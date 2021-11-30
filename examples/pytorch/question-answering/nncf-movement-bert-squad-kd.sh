
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
export WANDB_PROJECT="bert-squad-mvmt"
export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb

NEPOCH=5
RUNID=bert-squad-mvmt-${NEPOCH}eph-run4-kd
OUTDIR=/data1/vchua/jpq-nlp/phase2/${RUNID}
mkdir -p $OUTDIR

nohup python run_qa.py \
    --model_name_or_path csarron/bert-base-uncased-squad-v1 \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --learning_rate 3e-5 \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --save_steps 2500 \
    --nncf_config mvmt_config/nncf_bert_squad_mvmt_kd.json \
    --logging_steps 1 \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR 2>&1 | tee $OUTDIR/run.log &

    # --lr_scheduler_type cosine_with_restarts \
    # --warmup_ratio 0.05 \