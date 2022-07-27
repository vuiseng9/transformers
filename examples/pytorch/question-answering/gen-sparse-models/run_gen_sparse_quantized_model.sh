#!/usr/bin/env bash

WORKDIR=/data/vchua/dev/jpqd-alpha/transformers/examples/pytorch/question-answering
OUTROOT=gen-output/
CFGROOT=gen-sparse-models/

cd $WORKDIR
mkdir -p $OUTROOT

echo "[Info]: Creating bert-large-sparse-quantized/50%"
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-large-sparse-quantized/50% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-large_50%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-large-sparse-quantized/60%"
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-large-sparse-quantized/60% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-large_60%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-large-sparse-quantized/70%"
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-large-sparse-quantized/70% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-large_70%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-large-sparse-quantized/80%"
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-large-sparse-quantized/80% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-large_80%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-large-sparse-quantized/90%"
python run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-large-sparse-quantized/90% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-large_90%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-base-sparse-quantized/50%"
python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-base-sparse-quantized/50% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-base_50%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-base-sparse-quantized/60%"
python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-base-sparse-quantized/60% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-base_60%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-base-sparse-quantized/70%"
python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-base-sparse-quantized/70% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-base_70%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-base-sparse-quantized/80%"
python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-base-sparse-quantized/80% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-base_80%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating bert-base-sparse-quantized/90%"
python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/bert-base-sparse-quantized/90% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/bert-base_90%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating distilbert-sparse-quantized/50%"
python run_qa.py \
    --model_name_or_path distilbert-base-cased-distilled-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/distilbert-sparse-quantized/50% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/distilbert_50%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating distilbert-sparse-quantized/60%"
python run_qa.py \
    --model_name_or_path distilbert-base-cased-distilled-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/distilbert-sparse-quantized/60% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/distilbert_60%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating distilbert-sparse-quantized/70%"
python run_qa.py \
    --model_name_or_path distilbert-base-cased-distilled-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/distilbert-sparse-quantized/70% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/distilbert_70%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating distilbert-sparse-quantized/80%"
python run_qa.py \
    --model_name_or_path distilbert-base-cased-distilled-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/distilbert-sparse-quantized/80% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/distilbert_80%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

echo "[Info]: Creating distilbert-sparse-quantized/90%"
python run_qa.py \
    --model_name_or_path distilbert-base-cased-distilled-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTROOT/distilbert-sparse-quantized/90% \
    --nncf_config $CFGROOT/sweep_sparsity_nncfcfg/distilbert_90%_sparse_quantized.json \
    --overwrite_output_dir \
    --gen_onnx \

