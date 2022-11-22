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


# zeroshot clip
# 74.41
# +sigmoid 74.38
# +softmax 82.98
# +scale +softmax 83.16
# ------------------------------------------------------------------------------------------------------------------

###### reproduce
# bash main.sh oxford_flowers rn50 end 16 16 False oxford_flowers_reproduce2080ti

###### caption
# bash main_distill.sh voc2012_distill rn50 end 32 16 False voc2012_caption_distill 0  # 85.71
# bash main_distill.sh voc2012_distill rn50 end 32 16 False voc2012_caption_distill_add_single_label_sample 50000  # 85.88
# bash main_distill.sh voc2012_distill rn50 end 32 16 False voc2012_caption_distill_add_single_label_sample_2 3000   batch 128  85.82
# bash main_distill.sh voc2012_distill rn50 end 32 16 False voc2012_caption_distill_add_single_label_sample_b512 3000   batch 512  85.92

# bash main_distill.sh voc2012_distill rn50 end 32 16 False voc2012_caption_distill_add_single_label_sample_b512_asl 3000   batch 512 
# bash main_distill.sh voc2012_distill rn50 end 32 16 True voc2012_caption_distill_add_single_label_sample_b512_asl 3000   batch 512 


## 3090
# bash main_distill.sh voc2012_distill rn50 end 16 16 False voc2012_caption_distill_add_single_label_sample_b512_CE 0 6   batch 512  85.23
# ***** Dataset statistics *****
#   Dataset: VOC2012_distill
#   # classes: 20
#   # train_x: 118,287
#   # test: 5,823
# 旧版本的caption distill数据


# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE 4000 6   batch 512  85.18
# ***** Dataset statistics *****
#   Dataset: VOC2012_distill
#   # classes: 20
#   # train_x: 98,445
#   # test: 5,823
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try2 2000 4   batch 512    85.37   (0.019 0.26)
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try3 3000 5   batch 512    84.97  loss变大
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try4 1000 4   84.71

# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try5 1000 5   low data   84.51

# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try6 2000 4   batch 512  ablation data and baseline  85.06
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try8 2000 5   85.13
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try8_scale10 2000 7    scale 10

# valid训练，查看label归一化是否有用
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try9_valid_CE 2000 6       85.37
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try10_norm_label 2000 6    84.75

# 可学习的softmax temporature
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try11_learnable_temperature 2000 6    优化后超参为102，性能85.14

# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try7_BCE_singleP 2000 4    mAP score sigmoid: 47.63  softmax: 79.43  none: 58.71  ==========
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try8_BCE_singleP 2000 6    scale 10  mAP score sigmoid: 59.86  softmax: 81.97  none: 59.846488262184835
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try8_BCE_singleP_2 2000 6    随机种子不同，差别很大
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try8_BCE_singleP_shiftsigmoid 2000 7    



# CE
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try_valid_seed123 2000 6    100  
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try_valid_seed123_ft_t 2000 6   init temperature 100 better tuned to  4.6258(102)  4.6785(108)
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try_valid_seed123_ft_t_init70 2000 6   init temperature  70  log4.24   tuned to 4.3083(74.3)
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try_valid_seed123_t50 2000 6   fix t=50  
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try_valid_seed123_t70 2000 6   fix t=70

# voc2012_distill_3
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_b512_CE_try_valid_t70 2000 6   fix t=70

# voc2012_distill_v2
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_CE_try_valid_t70 2000 6   fix t=70

######################################################++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_b512_RL 2000 5 
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_RL 2000 5 
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_RL_traintemp10 2000 5 
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_RL_traintemp5 2000 6
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_RL_traintemp4 2000 5
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_RL_traintemp3 2000 6
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_RL_traintemp2 2000 6


# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_dual_3_v2_b512_RL_traintemp3 2000 6

# 在dual的情况下，使用sigmoid之后ranking，需要进一步验证梯度
# todo: bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_dual_3_v2_b512_RL_traintemp4 2000 6






# zclip：：：：：：：
# sigmoid 74                   111111111111111111
# softmax 83

# voc2012_caption_distill_new_data_b512_CE_try8 test, eval使用不同的激活函数：：：：：：
# mAP score sigmoid: 37.5521450645624
# mAP score softmax: 85.13415228318514
# mAP score none: 56.223696562228554
# mAP score: 56.223696562228554

# voc2012_caption_distill_new_data_b512_CE_try8_scale10
# 82.13
# 84.39
# 82.13

# voc2012_caption_distill_new_data_b512_CE_try7_BCE_singleP
# mAP score sigmoid: 47.83193562964742
# mAP score softmax: 79.43574061353124
# mAP score none: 58.71398575511814
# mAP score: 58.71398575511814

# voc2012_caption_distill_new_data_b512_CE_try8_BCE_singleP
# sigmoid 59.84              111111111111111111
# softmax 81.42
# none 59.84

# voc2012_caption_distill_new_data_b512_BCE_try3   dual prompt：：：：：
# mAP score sigmoid: 77.046168996377                    111111111111111111
# mAP score softmax: 83.6122891195991
# mAP score none: 77.046168996377
# mAP score: 77.04616899637










# ranking
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try1 2000 5       scale_ = 4  margin_ = 0.5     83.11
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try2 2000 3       scale_ = 10  margin_ = 0.5    82.15
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try3 2000 4       scale_ = 10  margin_ = 1      82.71
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try4 2000 5       scale_ = 10  margin_ = 2     83.2
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try5 2000 6       scale_ = 20  margin_ = 1     82.84
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try6 2000 5       scale_ = 4  margin_ = 1      83.79
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try7 2000 5       scale_ = 4  margin_ = 1.5    83.02
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_RL_try8 2000 6       scale_ = 4  margin_ = 2      83.12









# BCE  并且下面trainer都是Caption_distill_dual
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_BCE_try1 2000 4   batch 512   75.97   (-0.083 0.123)   for baseline dualp: (-0.09 0.029)
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_BCE_try2 2000 4   batch 512   lr 1e-5   not good
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_BCE_try3 2000 4   batch 512   lr 0.002  77.05
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_BCE_try4_scale10 2000 6   



# CE dual distill
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_CE_try1 2000 5    跟单prompt效果类似，但稍差  84.95

# old data BCE
# bash main_distill.sh voc2012_distill rn50 end 16 16 False voc2012_caption_distill_old_data_BCE 3000 6   batch 512    ~74


# dual+BCE 似乎在caption的情况下仍然效果不好
# 尝试基于 单prompt+mean+BCE结果
# TRAINER=Caption_distill
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_new_data_b512_mean_BCE 2000 5   batch 512  


# ============================================================================================================================================
# 上面是只用BCE的训练，效果不好，下面尝试结合CE
# 验证纯CE的结果
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_CE 2000 2    84.95

# CE + finetune temperature
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_CE_ft_t 2000 3   85.08

# CE + BCE相结合 取平均
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_CE+BCE_ft_t 2000 2     81.79   84.10
# CE + BCE相结合 0.9CE + 0.1bce
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_CE+BCE_ft_t_2 2000 3    84.22   84.64
# CE + BCE相结合 0.1CE + 0.9bce
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_CE+BCE_ft_t_3 2000 3    80.06


# 加载CE训练单prompt的model
# p+分支被冻结了，只使用CE loss 优化p-
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft 2000 3   85.26为起点   随机初始化p-
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_try2 2000 7   85.91为起点   随机初始化p-

# CE + BCE相结合 取平均
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_CE+BCE 2000 6   85.26为起点
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_CE+BCE_try2 2000 7   85.91为起点

# 只BCE
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_BCE 2000 6   85.91为起点
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_BCE_2 2000 6   85.91为起点  增加了测试设定 发现0.98 。。。

# merge_aux_v2  加入可学的lamda init = 0.98
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_BCE_3 2000 6   85.91为起点  收敛会更慢？  结果比较好，case study

# weighted BCE 在merge_aux_v2基础上加weightedBCE
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_w_BCE 2000 6   85.91为起点 
# 可学的lamda学到了0.6，有点离谱
# 固定lamda到0.98
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_w_BCE_2 2000 6   85.91为起点  

# 不加可学习lamda，只用weighted bce
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_w_BCE_3 2000 7   85.91为起点 

# 不加可学习lamda，只用focal BCE
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_freezeP_ft_w_BCE_4 2000 6   85.91为起点 


# 联合学习，不使用初始化
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_merge_branch 2000 6
# 还是要detach一下p+
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_merge_branch_2 2000 7

# add learnable lamda_
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_dual_new_data_b512_merge_branch_3 2000 6


# GCN
# bash main_distill.sh voc2012_distill_2 rn50 end 16 16 False voc2012_caption_distill_GCN 2000 6


# GT
# bash main_distill.sh voc2012_distill_gt rn50 end 16 16 False voc2012_caption_distill_gt 2000 6


# caption test



# COCO
# bash main_distill.sh coco2014_distill rn50 end 16 16 False coco2014_caption_distill 2000 6
# bash main_distill.sh coco2014_distill rn50 end 16 16 False coco2014_caption_distill_RL 2000  7

# bash main_distill.sh coco2014_distill rn50 end 16 16 False coco2014_caption_distill_dense_RL 2000 6
# bash main_distill.sh coco2014_distill rn50 end 16 16 False coco2014_caption_distill_v2_dense_RL_double 2000 6
# bash main_distill.sh coco2014_distill rn50 end 16 16 False coco2014_caption_distill_v3_dense_RL_double 2000 6
# bash main_distill.sh coco2014_distill rn50 end 16 16 False coco2014_caption_distill_v3_dense_RL_double_1500 1500 7

# bash main_distill.sh coco2014_distill rn50 end 32 16 False coco2014_caption_distill_v3_dense_RL_double_hp 2000 6



# NUSWIDE
# bash main_distill.sh nuswide_distill rn50 end 16 16 False nuswide_distill_RL 2000 7


# dense unify
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_dense_RL_traintemp4_3_5 2000 6
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_dense_RL_traintemp4_4 2000 6
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_dense_RL_traintemp4_4_addtextmask 2000 6     current best
# bash main_distill.sh voc2012_distill_3 rn50 end 16 16 False voc2012_caption_distill_3_v2_b512_dense_RL_traintemp4_4_CSC_double 2000 2




# before paper
# bash main_distill.sh voc2007_distill rn50 end 16 16 False voc2007_caption_distill_b512_dense_RL_addtextmask_ 2000 0  # 可学习的参数
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_b512_dense_RL_addtextmask_fixscale 2000 1  # 50 20
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_b512_dense_RL_addtextmask_fixscale_valid 2000 1  # 50 50
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_b512_dense_RL_addtextmask_fixscale_valid2 2000 6  
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_b512_dense_RL_addtextmask_fixscale_valid3_wofeatureensemble 2000 7  
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_b512_dense_RL_addtextmask_fixscale_valid4 2000 1  50 learn4.0 wofeatureensemble
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_b512_dense_RL_addtextmask_fixscale_valid5 2000 1  


# NUSWIDE
# bash main_distill.sh nuswide_distill rn50_fixscale end 16 16 False nuswide_distill_v2_RL_fixscale 2000 2
# bash main_distill.sh nuswide_distill rn50_fixscale end 16 16 False nuswide_distill_v2_RL_valid 2000 2    50 learn4.0 wofeatureensemble
# bash main_distill.sh nuswide_distill rn50_fixscale end 16 16 False nuswide_distill_v3_RL_valid 2000 1
# bash main_distill.sh nuswide_distill_limit rn50_fixscale end 16 16 False nuswide_distill_v3_limit_RL 2000 2   limit 1000
# bash main_distill.sh nuswide_distill_limit rn50_fixscale end 16 16 False nuswide_distill_v3_limit_RL_fixscale 2000 1    4.0  50
# bash main_distill.sh nuswide_distill_limit rn50_nuswide end 16 16 False nuswide_distill_v3_limit_RL_nuscfg 2000 1    best


# COCO
# bash main_distill.sh coco2014_distill rn50_coco2014 end 16 16 False coco2014_caption_distill_v4 2000 1   mean 64.89
# bash main_distill.sh coco2014_distill rn50_coco2014 end 16 16 False coco2014_caption_distill_v4_nofeatcap 0 1  mean 65.1  best


# ablation
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab0 2000 0   pure default
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab1000 2000 2
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab2000 2000 1
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab5000 2000 6
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab10000 2000 7
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab20000 2000 2
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab40000 2000 6
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf 2000 6    best
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_ab_pureword 2000 6



# ablation  LOSS fun
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf_asl 2000 0 asl
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf_asl_meanloss 2000 0 asl
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf_BCE 2000 2 sigmoid

# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf_asl_meanloss_lr002 2000 0 asl
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf_BCE_lr002 2000 1 sigmoid

## valid VOC
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf_BCE_lr002_valid 0 0 sigmoid
# VOC ASL
# bash main_distill.sh voc2007_distill rn50_fixscale end 16 16 False voc2007_caption_distill_abinf_ASL_lr002 0 1 asl

## COCO
# bash main_distill.sh coco2014_distill rn50_coco2014 end 16 16 False coco2014_caption_distill_v4_nofeatcap_BCE_2 0 6 sigmoid
# COCO ASL
# bash main_distill.sh coco2014_distill rn50_coco2014 end 16 16 False coco2014_caption_distill_v4_nofeatcap_ASL 0 2 asl

## NUS
# bash main_distill.sh nuswide_distill_limit rn50_nuswide end 16 16 False nuswide_distill_v3_limit_RL_nuscfg_BCE 0 7 sigmoid
# NUS ASL
# bash main_distill.sh nuswide_distill_limit rn50_nuswide end 16 16 False nuswide_distill_v3_limit_RL_nuscfg_ASL 0 6 asl
