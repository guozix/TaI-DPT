import pickle5 as pickle
import argparse

import numpy as np
import torch

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    # print('shape', targs.shape)
    # print(preds.shape)
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default="")
    parser.add_argument("--file2", type=str, default="")
    parser.add_argument("--ensemble_rate", type=float, default=0.5)
    args = parser.parse_args()
    
    with open(args.file1, 'rb') as f:
        info = pickle.load(f)
    targets, preds, preds_aux = info
    targets = (targets == 1).astype('int')
    
    preds_merge = (preds.cpu().numpy() * 0.5 + preds_aux.cpu().numpy() * 0.5) 
    mAP_score1 = mAP(targets, preds_merge)
    print("mAP score TaI-DPT:", mAP_score1)
    
    #
    with open(args.file2, 'rb') as f:
        info = pickle.load(f)
    targets_, preds2, preds_aux2 = info
    
    preds2_ = preds_aux2.cpu().numpy()
    mAP_score2 = mAP(targets, preds2_)
    print("mAP score:", mAP_score2)
    
    #
    tmp1 = preds_merge
    tmp2 = preds2_
    tmp1_ = (tmp1 - tmp1.min()) / (tmp1.max() - tmp1.min())
    tmp2_ = (tmp2 - tmp2.min()) / (tmp2.max() - tmp2.min())
        
    tmp = (tmp1_ * (1-args.ensemble_rate) + tmp2_ * args.ensemble_rate)  # (preds_sigmoid + preds_softmax) / 2
    mAP_score_ = mAP(targets, tmp) 
    print("mAP score ensembled", mAP_score_)

main()
