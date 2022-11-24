#!/bin/bash

cd ..

# custom config
# DATA=/home/weiyuxiang/gzx/VOS/datas
DATA=/home/qiangwenjie/datasets
# TRAINER=Caption
# TRAINER=Caption_distill
# TRAINER=Caption_distill_dual
TRAINER=Caption_distill_double

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
run_ID=$7
trainset_sample=$8
loss=${10}

export CUDA_VISIBLE_DEVICES=$9

for SEED in 1 2 3 # 4 5 6 7 8 9
do
    DIR=output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job andsave the output to ${DIR}"
        # python train.py \  CUDA_VISIBLE_DEVICES=1
        python train_caption.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.Caption.N_CTX ${NCTX} \
        TRAINER.Caption.CSC ${CSC} \
        TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SAMPLE ${trainset_sample} \
        TRAIN.LOSSFUNC ${loss}
    fi
done
