#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778
export HF_HOME=/data4/vchua/cache/huggingface
# ---------------------------------------------------------------------------------------------
export WANDB_PROJECT="wav2vec2 (sixer)"
export CUDA_VISIBLE_DEVICES=0,1
readarray -td, gpuarray <<< $CUDA_VISIBLE_DEVICES
declare -p gpuarray
NGPU=${#gpuarray[@]}

NEPOCH=40
BS_TRAIN=8
BS_EVAL=16
RUNID=wav2vec2-base-ls100h-${NEPOCH}eph-${NGPU}gpu-each-${BS_TRAIN}bs

CONDAROOT=/data5/vchua/miniconda3
CONDAENV=may23-w2v2-asr

OUTROOT=/data5/vchua/run/$CONDAENV/
WORKDIR=/data5/vchua/dev/$CONDAENV/transformers/examples/pytorch/speech-recognition/

# ---------------------------------------------------------------------------------------------
OUTDIR=$OUTROOT/$RUNID

# override label if in dryrun mode
if [[ $1 == "dryrun" ]]; then
    OUTDIR=$OUTROOT/dryrun-${RUNID}
    RUNID=dryrun-${RUNID}
fi

mkdir -p $OUTDIR
cd $WORKDIR

# cmd="
python run_speech_recognition_ctc.py \
        --dataset_name="librispeech_asr" \
        --model_name_or_path="facebook/wav2vec2-base" \
        --overwrite_output_dir \
		--run_name $RUNID \
        --output_dir=$OUTDIR \
        --train_split_name="train.clean.100" \
        --eval_split_name="validation.clean" \
        --num_train_epochs=$NEPOCH \
        --per_device_train_batch_size=$BS_TRAIN \
        --per_device_eval_batch_size=$BS_EVAL \
        --gradient_accumulation_steps="4" \
        --weight_decay="0.000" \
        --learning_rate="1e-4" \
        --warmup_steps="3000" \
        --evaluation_strategy="steps" \
        --text_column_name="text" \
        --save_steps="500" \
        --eval_steps="100" \
        --logging_steps="50" \
        --layerdrop="0.1" \
        --save_total_limit="3" \
        --freeze_feature_encoder \
        --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
        --fp16 \
        --group_by_length \
        --do_train --do_eval \
        --max_steps 12000

        # --push_to_hub \

# if [[ $1 == "local" ]]; then
#     echo "${cmd}" > $OUTDIR/run.log
#     echo "### End of CMD ---" >> $OUTDIR/run.log
#     cmd="nohup ${cmd}"
#     eval $cmd >> $OUTDIR/run.log 2>&1 &
#     echo "logpath: $OUTDIR/run.log"
# elif [[ $1 == "dryrun" ]]; then
#     echo "[INFO: dryrun, add --max_steps 25 to cli"
#     cmd="${cmd} --max_steps 25"
#     echo "${cmd}" > $OUTDIR/dryrun.log
#     echo "### End of CMD ---" >> $OUTDIR/dryrun.log
#     eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
#     echo "logpath: $OUTDIR/dryrun.log"
# else
#     source $CONDAROOT/etc/profile.d/conda.sh
#     conda activate ${CONDAENV}
#     eval $cmd
# fi