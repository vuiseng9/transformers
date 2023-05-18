#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778
export HF_HOME=/data4/vchua/cache/huggingface
# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="sparse-attn-act ($(hostname))"
export CUDA_VISIBLE_DEVICES=0

CONDAROOT=/data5/vchua/miniconda3
CONDAENV=apr23-optimum

NEPOCH=2
LR=3e-5
RUNID=exp2-softmax-ft-squad-bert-b-${NEPOCH}eph-lr${LR}

OUTROOT=/data5/vchua/run/$CONDAENV/sparse-attn-act
WORKDIR=/data5/vchua/dev/$CONDAENV/transformers/examples/pytorch/question-answering
# ---------------------------------------------------------------------------------------------

OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR

cmd="
python run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_eval \
    --fp16 \
    --softmax_exp2 \
    --do_train \
    --learning_rate $LR \
    --num_train_epochs $NEPOCH \
    --per_device_eval_batch_size 128 \
    --per_device_train_batch_size 16 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --logging_steps 1 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 1000 \
    --save_total_limit 10 \
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



