from calendar import c
import os
from os.path import join
from re import L
from numpy import dtype
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import numpy as np
import json
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights
from trainers.coop import load_clip_to_cpu

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .data_helpers import *

object_categories = nuswide_object_categories
classname_synonyms = nuswide_classname_synonyms


@DATASET_REGISTRY.register()
class nuswide_partial(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'NUSWIDE'
        cls_num = 81
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Flickr")
        self.cls_name_list = self.read_name_list(join(self.dataset_dir, 'Concepts81.txt'), False)
        self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/TrainImagelist.txt'), False)
        self.im_name_list_test = self.read_name_list(join(self.dataset_dir, 'ImageList/TestImagelist.txt'), False)
        print('NUS-WIDE total {} images, test {} images. '.format(len(self.im_name_list), len(self.im_name_list_test)))

        path_labels = os.path.join(self.dataset_dir, 'TrainTestLabels')
        num_classes = len(object_categories)
        
        partial_prob = cfg.DATASET.partial_prob
        print('Creating NUSWIDE partial prob:', partial_prob)
        
        def changeLabelProportion(labels, label_proportion):
            mask = np.random.random(labels.shape)
            mask[mask < label_proportion] = 1
            mask[mask < 1] = 0
            label = mask * labels
            assert label.shape == labels.shape
            return label

        train_labels = [] # torch.ones((len(self.im_name_list_test), num_classes))
        for i in tqdm(range(num_classes)):
            file_ = os.path.join(path_labels, 'Labels_'+object_categories[i]+'_Train.txt')
            cls_labels = []
            with open(file_, 'r') as f:
                for j, line in enumerate(f):
                    tmp = line.strip()
                    cls_labels.append(1 if int(tmp)>0 else -1)
            train_labels.append(torch.tensor(cls_labels, dtype=torch.long))
        train_labels = torch.stack(train_labels, dim=1)
        
        train_labels_partial = changeLabelProportion(train_labels.numpy(), partial_prob)
        
        train = []
        for i, name in enumerate(self.im_name_list):
            item_ = Datum(impath=self.image_dir + '/' + '/'.join(name.split('\\')), label=train_labels_partial[i], classname='')
            train.append(item_)
            
        ############################################################
        test_labels = [] # torch.ones((len(self.im_name_list_test), num_classes))
        for i in tqdm(range(num_classes)):
            file_ = os.path.join(path_labels, 'Labels_'+object_categories[i]+'_Test.txt')
            cls_labels = []
            with open(file_, 'r') as f:
                for j, line in enumerate(f):
                    tmp = line.strip()
                    cls_labels.append(int(tmp))
            test_labels.append(torch.tensor(cls_labels, dtype=torch.long))
        test_labels = torch.stack(test_labels, dim=1)
        
        test = []
        for i, name in enumerate(self.im_name_list_test):
            item_ = Datum(impath=self.image_dir + '/' + '/'.join(name.split('\\')), label=test_labels[i], classname='')
            test.append(item_)

        super().__init__(train_x=train, val=test[0::60], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})

    def read_name_list(self, path, if_split=True):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                if if_split:
                    tmp = line.strip().split(' ')
                    ret.append(tmp[0])
                else:
                    tmp = line.strip()
                    ret.append(tmp)
        return ret

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test

