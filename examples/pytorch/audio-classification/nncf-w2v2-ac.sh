#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="w2v2opt-ks ($(hostname))"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=12
TEMPERATURE=2
ALPHA=0.9
RUNID=run33.6-w2v2b-ks-mvmt-bt-${NEPOCH}eph-warmup1-6-r0.10-t${TEMPERATURE}-alpha${ALPHA}
NNCFCFG=/nvme1/vchua/dev/p3-w2v2/transformers/examples/pytorch/audio-classification/cfg-nncf/nncf_w2v2b_ks_mvmt.json
OUTROOT=/nvme1/vchua/run/p3-w2v2/w2v2b-ks/nncf-mvmt


WORKDIR=/nvme1/vchua/dev/p3-w2v2/transformers/examples/pytorch/audio-classification

CONDAROOT=/nvme1/vchua/miniconda3
CONDAENV=p3-w2v2
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR
    # --lr_scheduler_type cosine_with_restarts \

cmd="
python run_audio_classification.py \
    --teacher anton-l/wav2vec2-base-ft-keyword-spotting \
    --teacher_ratio $ALPHA \
    --distill_temp $TEMPERATURE \
    --model_name_or_path anton-l/wav2vec2-base-ft-keyword-spotting \
    --dataset_name superb \
    --dataset_config_name ks \
    --remove_unused_columns False \
    --do_eval \
    --do_train \
    --nncf_config $NNCFCFG \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs $NEPOCH \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --seed 0 \
    --run_name $RUNID \
    --output_dir $OUTDIR \
    --overwrite_output_dir
"
    # --save_total_limit 3 \

if [[ $1 == "local" ]]; then
    echo "${cmd}" > $OUTDIR/run.log
    echo "### End of CMD ---" >> $OUTDIR/run.log
    cmd="nohup ${cmd}"
    eval $cmd >> $OUTDIR/run.log 2>&1 &
    echo "logpath: $OUTDIR/run.log"
elif [[ $1 == "dryrun" ]]; then
    echo "[INFO: dryrun, add --max_steps 25 to cli"
    cmd="${cmd} --max_steps 25"
    echo "${cmd}" > $OUTDIR/dryrun.log
    echo "### End of CMD ---" >> $OUTDIR/dryrun.log
    eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
    echo "logpath: $OUTDIR/dryrun.log"
else
    source $CONDAROOT/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    eval $cmd
fi