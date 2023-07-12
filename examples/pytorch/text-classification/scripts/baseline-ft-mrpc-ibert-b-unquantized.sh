#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778
export HF_HOME=/data4/vchua/cache/huggingface
# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="ibert (sixer)"
export CUDA_VISIBLE_DEVICES=0

CONDAROOT=/data5/vchua/miniconda3
CONDAENV=ibert-alt-softmax

NEPOCH=5
LR=2e-5
TASK=mrpc
RUNID=baseline-ft-${TASK}-IRoberta-b-${NEPOCH}eph-lr${LR}


OUTROOT=/data5/vchua/run/$CONDAENV/$TASK
WORKDIR=/data5/vchua/dev/$CONDAENV/transformers/examples/pytorch/text-classification
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR

    # --fp16 \
    # --eval_steps 250 \
    # --save_steps 1000 \
    # --warmup_ratio 0.06 \
    # --weight_decay 0.1 \
    # --save_total_limit 3 \
cmd="
python run_glue.py \
    --model_name_or_path kssteven/ibert-roberta-base \
    --task_name $TASK \
    --do_eval \
    --do_train \
    --learning_rate $LR \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --max_seq_length 128 \
    --logging_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir \
    --run_name $RUNID \
    --output_dir $OUTDIR \
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



