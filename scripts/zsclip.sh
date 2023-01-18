#!/bin/bash

cd ..

# custom config
DATA=/home/qiangwenjie/datasets
TRAINER=ZeroshotCLIP_dense

DATASET=$1
CFG=$2  # config file

export CUDA_VISIBLE_DEVICES=0

DIR=output/evaluation/${TRAINER}/${DATASET}/${CFG}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job andsave the output to ${DIR}"
    python train_caption.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/zsclip/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only
fi

# bash zsclip.sh voc2007_distill rn50
# bash zsclip.sh coco2014_distill rn50
# bash zsclip.sh nuswide_distill_limit rn50
