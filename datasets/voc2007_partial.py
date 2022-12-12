import os
from os.path import join
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import torch.utils.data as data
import json
from tqdm import tqdm

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

import sys
# sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from PIL import Image
import xml.dom.minidom
from xml.dom.minidom import parse

from .data_helpers import *

object_categories = voc_object_categories
classname_synonyms = voc_classname_synonyms

def read_labels(path_labels):
    file = path_labels
    labels = []
    with open(file, 'r') as f:
        for line in f:
            tmp = list(map(int, line.strip().split(',')))
            labels.append(torch.tensor(tmp, dtype=torch.long))
    return labels


def read_name_list(path):
    ret = []
    with open(path, 'r') as f:
        for line in f:
            tmp = line.strip()
            ret.append(tmp)
    return ret


@DATASET_REGISTRY.register()
class VOC2007_partial(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'VOCdevkit/VOC2007'
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "JPEGImages")
        self.im_name_list = read_name_list(join(self.dataset_dir, 'ImageSets/Main/trainval.txt'))
        self.im_name_list_val = read_name_list(join(self.dataset_dir, 'ImageSets/Main/test.txt'))
        print('VOC2007 train total {} images, test total {} images. '.format(len(self.im_name_list), len(self.im_name_list_val)))

        caption_feat_root = os.getcwd()
        partial_prob = cfg.DATASET.partial_prob
        print('Loading', 'partial/VOC2007/partial-labels/train_proportion_{}.txt'.format(partial_prob))
        
        train_labels = read_labels(join(caption_feat_root, 'partial/VOC2007/partial-labels/train_proportion_{}.txt'.format(partial_prob)))
        test_labels = read_labels(join(caption_feat_root, 'partial/VOC2007/partial-labels/val.txt'))
        
        train = []
        for i, name in enumerate(self.im_name_list):
            item_ = Datum(impath=self.image_dir+'/{}.jpg'.format(name), label=train_labels[i], classname='')
            train.append(item_)

        test = []
        for i, name in enumerate(self.im_name_list_val):
            item_ = Datum(impath=self.image_dir+'/{}.jpg'.format(name), label=test_labels[i], classname='')
            test.append(item_)
    
        super().__init__(train_x=train, val=test, test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
