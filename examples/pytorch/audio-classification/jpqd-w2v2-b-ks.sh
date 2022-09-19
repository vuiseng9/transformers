#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="w2v2opt-ks (dgx4)"
export CUDA_VISIBLE_DEVICES=0

NEPOCH=15
BS=64
GA=2
LR=3e-5
ALPHA=0.9
RUNID=jpqd-ft-w2v2-b-ks-bs${BS}-${NEPOCH}eph-$LR-r0.50-sigmoid-wu5-10

NNCFCFG=/data2/vchua/dev/w2v2-ion/transformers/examples/pytorch/audio-classification/nncfcfg/jpqd-w2v2-b-ks.json
OUTROOT=/data1/vchua/run/w2v2-ion/w2v2b-ks/

WORKDIR=/data2/vchua/dev/w2v2-ion/transformers/examples/pytorch/audio-classification
CONDAROOT=/data1/vchua/miniconda3
CONDAENV=w2v2-ion
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
    # --distill_temp $TEMPERATURE \
    # --load_best_model_at_end True \
    # --model_name_or_path facebook/wav2vec2-base \
    # --save_total_limit 3 \

cmd="
python run_audio_classification.py \
    --model_name_or_path facebook/wav2vec2-base \
    --teacher anton-l/wav2vec2-base-ft-keyword-spotting \
    --teacher_ratio $ALPHA \
    --dataset_name superb \
    --dataset_config_name ks \
    --remove_unused_columns False \
    --do_eval \
    --do_train \
    --nncf_config $NNCFCFG \
    --learning_rate $LR \
    --max_length_seconds 1 \
    --attention_mask False \
    --num_train_epochs $NEPOCH \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps $GA \
    --per_device_eval_batch_size $BS \
    --dataloader_num_workers 8 \
    --logging_strategy steps \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --seed 0 \
    --run_name $RUNID \
    --output_dir $OUTDIR \
    --overwrite_output_dir
"

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