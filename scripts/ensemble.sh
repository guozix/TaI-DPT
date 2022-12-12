#!/bin/bash

cd ..

# custom config
DATA=/home/qiangwenjie/datasets
TRAINER=Caption_distill_double

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
CSC=$5  # class-specific context (False or True)
run_ID=$6
res_dir=$7
model1=$8

export CUDA_VISIBLE_DEVICES=6

DIR=output/${run_ID}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}
echo "Run this job andsave the output to ${DIR}"
python train_caption.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${model1} \
--eval-only \
TRAINER.Caption.N_CTX ${NCTX} \
TRAINER.Caption.CSC ${CSC} \
TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP} \
TEST.SAVE_PREDS ${res_dir}


#
TRAINER2=Caption_dual

DATASET2=$9
CFG2=${10}  # config file
CTP2=${11}  # class token position (end or middle)
NCTX2=${12}  # number of context tokens
CSC2=${13}  # class-specific context (False or True)
run_ID2=${14}
res_dir2=${15}
model2=${16}

DIR2=output/${run_ID2}/${TRAINER2}/${CFG2}/nctx${NCTX2}_csc${CSC2}_ctp${CTP2}
echo "Run this job andsave the output to ${DIR2}"
python train_caption.py \
--root ${DATA} \
--trainer ${TRAINER2} \
--dataset-config-file configs/datasets/${DATASET2}.yaml \
--config-file configs/trainers/${TRAINER2}/${CFG2}.yaml \
--output-dir ${DIR2} \
--model-dir ${model2} \
--eval-only \
TRAINER.Caption.N_CTX ${NCTX2} \
TRAINER.Caption.CSC ${CSC2} \
TRAINER.Caption.CLASS_TOKEN_POSITION ${CTP2} \
TEST.SAVE_PREDS ${res_dir2}

ensemble_rate=${17}
python ensemble_score.py --file1 ${res_dir} --file2 ${res_dir2} --ensemble_rate ${ensemble_rate}

# bash ensemble.sh voc2007_distill rn50_voc2007 end 16 False voc2007_caption_e tmp1.pkl \
#  output/voc2007_caption/Caption_distill_double/rn50_voc2007/nctx16_cscFalse_ctpend/seed1 \
# voc2007_partial rn101 end 16 True voc2007_partial_dualcoop_448_CSC_p0_5_e tmp2.pkl \
#  output/voc2007_partial_dualcoop_448_CSC_p0_5/Caption_dual/rn101/nctx16_cscTrue_ctpend/seed1 0.9


