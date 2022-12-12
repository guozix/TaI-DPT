import os
from os.path import join
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
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
import torch
import torch.utils.data as data

from pycocotools.coco import COCO

from .data_helpers import *

object_categories = coco_object_categories
classname_synonyms = coco_classname_synonyms

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
class COCO2014_partial(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'COCO'
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        partial_prob = cfg.DATASET.partial_prob
        print('Creating COCO2014 partial prob:', partial_prob)

        coco2014_train = os.path.join(self.dataset_dir, "annotations/instances_train2014.json")
        self.coco_train = COCO(coco2014_train)
        self.ids_train = self.coco_train.getImgIds()
        
        ## ==============================================================================
        categories = self.coco_train.loadCats(self.coco_train.getCatIds())
        categories.sort(key=lambda x: x['id'])
        # print(categories)

        classes = {}
        coco_labels = {}
        coco_labels_inverse = {}
        for c in categories:
            coco_labels[len(classes)] = c['id']
            coco_labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)

        # also load the reverse (label -> name)
        labels = {}
        for key, value in classes.items():
            labels[value] = key
        
        num_cls = len(object_categories)
        ## ==============================================================================
              
        def load_annotations(coco_, img_idlist, image_index, filter_tiny=True):
            # get ground truth annotations
            tmp_id = image_index if (img_idlist is None) else img_idlist[image_index]
            annotations_ids = coco_.getAnnIds(imgIds=tmp_id, iscrowd=False)
            annotations = []

            # some images appear to miss annotations (like image with id 257034)
            if len(annotations_ids) == 0:
                return annotations

            # parse annotations
            coco_annotations = coco_.loadAnns(annotations_ids)
            for idx, a in enumerate(coco_annotations):
                # some annotations have basically no width / height, skip them
                if filter_tiny and (a['bbox'][2] < 1 or a['bbox'][3] < 1):
                    continue
                annotations += [coco_label_to_label(a['category_id'])]

            return annotations

        def coco_label_to_label(coco_label):
            return coco_labels_inverse[coco_label]

        def label_to_coco_label(label):
            return coco_labels[label]

        def labels_list_to_1hot_partial(labels_list, class_num):
            labels_1hot = np.ones(class_num, dtype=np.float32) * (-1)
            labels_1hot[labels_list] = 1
            return labels_1hot

        
        def changeLabelProportion(labels, label_proportion):
            mask = np.random.random(labels.shape)
            mask[mask < label_proportion] = 1
            mask[mask < 1] = 0
            label = mask * labels
            assert label.shape == labels.shape
            return label
        
        train_labels = []
        for idx, imgid in enumerate(self.ids_train):
            label_tmp = load_annotations(self.coco_train, None, imgid)
            label_tmp = labels_list_to_1hot_partial(label_tmp, num_cls)
            train_labels.append(label_tmp)
        train_labels = np.stack(train_labels, axis=0)
        print('train_labels.shape =', train_labels.shape)
        
        self.train_labels = changeLabelProportion(train_labels, partial_prob)

        train = []
        for idx, imgid in enumerate(self.ids_train):
            img_dir = self.dataset_dir + '/train2014/{}'.format(self.coco_train.loadImgs(imgid)[0]['file_name'])
            item_ = Datum(impath=img_dir, label=self.train_labels[idx], classname='')
            train.append(item_)

        ######################################################
        coco2014_val = os.path.join(self.dataset_dir, "annotations/instances_val2014.json")
        self.coco_val = COCO(coco2014_val)
        self.ids_val = self.coco_val.getImgIds()
        
        test = []
        for idx, imgid in enumerate(self.ids_val):
            img_dir = self.dataset_dir + '/val2014/{}'.format(self.coco_val.loadImgs(imgid)[0]['file_name'])
            labels_ = labels_list_to_1hot_partial(load_annotations(self.coco_val, None, imgid, filter_tiny=False), num_cls)
            item_ = Datum(impath=img_dir, label=labels_, classname='')
            test.append(item_)
        
        super().__init__(train_x=train, val=test[0::20], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
        