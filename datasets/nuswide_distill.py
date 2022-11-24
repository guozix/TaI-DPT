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
import json
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights
from trainers.coop import load_clip_to_cpu

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .data_helpers import *

prompt_template = "a photo of {}."

classname_synonyms =[
    ['airport', 'air port', 'air field'],
    ['animal'],
    ['beach', 'plage', 'coast'],
    ['bear'],
    ['birds', 'bird'],
    ['boats', 'boat'],
    ['book'],
    ['bridge'],
    ['buildings', 'building'],
    ['cars', 'car'],
    ['castle'],
    ['cat', 'kitty'],
    ['cityscape'],
    ['clouds', 'cloud'],
    ['computer'],
    ['coral'],
    ['cow'],
    ['dancing', 'dance'],
    ['dog', 'pup', 'puppy', 'doggy'],
    ['earthquake'],
    ['elk'],
    ['fire'],
    ['fish'],
    ['flags'],
    ['flowers'],
    ['food'],
    ['fox'],
    ['frost'],
    ['garden'],
    ['glacier'],
    ['grass'],
    ['harbor', 'port', 'harbour'],
    ['horses', 'horse'],
    ['house'],
    ['lake'],
    ['leaf'],
    ['map'],
    ['military', 'army' , 'troops'],
    ['moon'],
    ['mountain', 'hill'],
    ['nighttime', 'night time', 'night'],
    ['ocean', 'sea'],
    ['person', 'human', 'people', 'man', 'woman', 'passenger'],
    ['plane', 'aeroplane', "air craft", "jet", "air plane"],
    ['plants'],
    ['police'],
    ['protest'],
    ['railroad', 'rail road', 'rail way'],
    ['rainbow'],
    ['reflection'],
    ['road', 'path', 'way'],
    ['rocks', 'rock'],
    ['running', 'run'],
    ['sand'],
    ['sign'],
    ['sky'],
    ['snow'],
    ['soccer', 'football'],
    ['sports', 'sport'],
    ['statue'],
    ['street'],
    ['sun'],
    ['sunset'],
    ['surf'],
    ['swimmers', 'swimmer', 'swimming'],
    ['tattoo', 'tattooing'],
    ['temple'],
    ['tiger'],
    ['tower'],
    ['town'],
    ['toy'],
    ['train'],
    ['tree'],
    ['valley'],
    ['vehicle'],
    ['water'],
    ['waterfall'],
    ['wedding'],
    ['whales', 'whale'],
    ['window'],
    ['zebra'],
]

clsname2idx_ = {}
nameset_compound = set()
nameset = set()
for idx, synset in enumerate(classname_synonyms):
    for n in synset:
        clsname2idx_[n] = idx

        if ' ' in n:
            nameset_compound.add(n)
            m = n.replace(' ', '')
            clsname2idx_[m] = idx
            nameset.add(m)
        else:
            nameset.add(n)


object_categories = [syn[0] for syn in classname_synonyms]


@DATASET_REGISTRY.register()
class nuswide_distill(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'NUSWIDE'
        cls_num = 81
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Flickr")
        self.cls_name_list = self.read_name_list(join(self.dataset_dir, 'Concepts81.txt'), False)
        self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/Imagelist.txt'), False)
        # self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/Imagelist.txt'), False)
        self.im_name_list_test = self.read_name_list(join(self.dataset_dir, 'ImageList/TestImagelist.txt'), False)
        print('NUS-WIDE total {} images, test {} images. '.format(len(self.im_name_list), len(self.im_name_list_test)))

        caption_feat_root = '/home/qiangwenjie/gzx/CoOp-main'  # '/home/weiyuxiang/gzx/CoOp-main'


        # if os.path.exists(join(caption_feat_root, 'nuswide_train.pkl')):
        #     with open(join(caption_feat_root, 'nuswide_train.pkl'), 'rb') as f:
        #         train = pickle.load(f)
                
        #     test = []
        #     for i, name in enumerate(self.im_name_list_test):
        #         item_ = Datum(impath=self.image_dir + '/' + '/'.join(name.split('\\')), label=test_labels[i], classname='')
        #         test.append(item_)

        #     super().__init__(train_x=train, val=None, test=test, \
        #         num_classes=len(object_categories), classnames=object_categories, \
        #         lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
        
        # ##### if exists, just read label
        if os.path.exists(join(caption_feat_root, 'nuswide_cls_word_based_caption_labels_v3.pkl')):

            with open(join(caption_feat_root, 'nuswide_cls_word_based_caption_labels_v3.pkl'), 'rb') as f:
                word_based_caption = pickle.load(f)
            with open(join(caption_feat_root, 'nuswide_cls_capid_filterword_empty_v3.pkl'), 'rb') as f:
                capid_empty_filter = pickle.load(f)
            with open(join(caption_feat_root, 'nuswide_cls_feature_based_caption_selected.pkl'), 'rb') as f:  #  nuswide_cls_feature_based_caption_selected_1w_percls_v1.pkl
                feature_based_caption = pickle.load(f)
        
        else:
            pass

        
        # 数据集是tokenized形式的prompt，后续输入模型提取text feature
        with open(join(caption_feat_root, 'all_caption_tokenized_open_images.pkl'), 'rb') as f:
            prompts = pickle.load(f)
        print(prompts.shape)


        # sample_capid_inverse_idx = {}
        # for i, j in enumerate(sample_capid):
        #     sample_capid_inverse_idx[j] = i

        # =============================================================
        train = []
        for capid in word_based_caption:
            i = capid - 1
            item_ = (prompts[i], word_based_caption[capid])
            train.append(item_)
        print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(word_based_caption)))
        # 暂且认为能筛选出caption的都是好的caption，尽可能保留作为训练数据
        
        # # 保存共现频率
        # co_labels = torch.stack([k[1] for k in train])
        # count_prob = torch.zeros((cls_num, cls_num)).float()
        # for i in range(cls_num):
        #     idx = co_labels[:, i] == 1
        #     cur_t = co_labels[idx]
        #     num_ = cur_t.shape[0]
        #     cur_t = cur_t.sum(axis=0)
        #     count_prob[i] = (cur_t / num_)
        # torch.save(count_prob, 'count_prob_voc2012_caption_v2.pkl')

        # -------------------------------------------------------------
        # train = []
        # capid_list = random.sample(list(word_based_caption.keys()), 500)
        # for capid in capid_list:
        #     i = sample_capid_inverse_idx[capid]
        #     item_ = (prompts[i], torch.tensor(word_based_caption[capid]))
        #     train.append(item_)
        # print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(word_based_caption)))
        # =============================================================


        # if cfg.DATASET.SAMPLE > 0:
        #     seed = cfg.SEED
        #     sample_train_num = cfg.DATASET.SAMPLE
        #     print("Sampling {} training images with seed {}...".format(sample_train_num, seed))
        #     train = random.sample(train, sample_train_num)


        # =============================================================
        # 这里cfg.DATASET.SAMPLE该参数先用做挑选单label的caption数量
        # feature_based_caption    clsidx 2 capid
        cur_count = [0] * cls_num
        for i in feature_based_caption:
            label = [0] * cls_num
            label[i] = 1
            for capidx in feature_based_caption[i][:cfg.DATASET.SAMPLE]:
                if (capidx + 1) in capid_empty_filter:
                    idx = capidx
                    train.append((prompts[idx], label))
                    cur_count[i] += 1
        print("===== Caption Distill Data: nums of class feature filtered caption  =====\n{}".format(cur_count))
        print("===== total: {}  =====\n".format(torch.tensor(cur_count).sum()))

        # -------------------------------------------------------------
        
        # cur_count = [0] * 20
        # for i in feature_based_caption:
        #     label = [0] * 20
        #     label[i] = 1
        #     for capid in feature_based_caption[i][:10]:   # low data mode
        #         if capid in capid_empty_filter:
        #             idx = sample_capid_inverse_idx[capid]
        #             train.append((prompts[idx], torch.tensor(label)))
        #             cur_count[i] += 1
        # print("===== Caption Distill Data: nums of class feature filtered caption  =====\n{}".format(cur_count))

        # =============================================================
        
        # default template
        default_prompt_num = 10 # 1  100
        for i in range(cls_num):
            label = [0] * cls_num
            label[i] = 1
            tmp_p = clip.tokenize(prompt_template.format(object_categories[i]))[0]
            for j_ in range(default_prompt_num-1):
                train.append((tmp_p, torch.tensor(label)))
            
            for cur_temp in IMAGENET_TEMPLATES:
                tmp_p = clip.tokenize(cur_temp.format(object_categories[i]))[0]
                train.append((tmp_p, torch.tensor(label)))

        ############################
        ## test data
        
        # def read_image_label(file):
        #     print('[dataset] read ' + file)
        #     data_ = dict()
        #     with open(file, 'r') as f:
        #         for line in f:
        #             tmp = line.strip().split(' ')
        #             name = tmp[0]
        #             label = int(tmp[-1])
        #             data_[name] = label
        #     return data_


        # def read_object_labels(path, phase):
        #     return labeled_data

        # test_data_imname2label = read_object_labels(self.dataset_dir, phase='val')
        
        path_labels = os.path.join(self.dataset_dir, 'TrainTestLabels')
        # labeled_data = dict()
        num_classes = len(object_categories)

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

        super().__init__(train_x=train, val=test[0::20], test=test, \
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


@DATASET_REGISTRY.register()
class nuswide_distill_limit(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'NUSWIDE'
        cls_num = 81
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "Flickr")
        self.cls_name_list = self.read_name_list(join(self.dataset_dir, 'Concepts81.txt'), False)
        self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/Imagelist.txt'), False)
        # self.im_name_list = self.read_name_list(join(self.dataset_dir, 'ImageList/Imagelist.txt'), False)
        self.im_name_list_test = self.read_name_list(join(self.dataset_dir, 'ImageList/TestImagelist.txt'), False)
        print('NUS-WIDE total {} images, test {} images. '.format(len(self.im_name_list), len(self.im_name_list_test)))

        caption_feat_root = '/home/qiangwenjie/gzx/CoOp-main'  # '/home/weiyuxiang/gzx/CoOp-main'


        # if os.path.exists(join(caption_feat_root, 'nuswide_train.pkl')):
        #     with open(join(caption_feat_root, 'nuswide_train.pkl'), 'rb') as f:
        #         train = pickle.load(f)
                
        #     test = []
        #     for i, name in enumerate(self.im_name_list_test):
        #         item_ = Datum(impath=self.image_dir + '/' + '/'.join(name.split('\\')), label=test_labels[i], classname='')
        #         test.append(item_)

        #     super().__init__(train_x=train, val=None, test=test, \
        #         num_classes=len(object_categories), classnames=object_categories, \
        #         lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
        
        # ##### if exists, just read label
        if os.path.exists(join(caption_feat_root, 'nuswide_cls_word_based_caption_labels_v3.pkl')):

            with open(join(caption_feat_root, 'nuswide_cls_word_based_caption_labels_v3.pkl'), 'rb') as f:
                word_based_caption = pickle.load(f)
            with open(join(caption_feat_root, 'nuswide_cls_capid_filterword_empty_v3.pkl'), 'rb') as f:
                capid_empty_filter = pickle.load(f)
            with open(join(caption_feat_root, 'nuswide_cls_feature_based_caption_selected.pkl'), 'rb') as f:  #  nuswide_cls_feature_based_caption_selected_1w_percls_v1.pkl
                feature_based_caption = pickle.load(f)
        
        else:
            pass

        
        # 数据集是tokenized形式的prompt，后续输入模型提取text feature
        with open(join(caption_feat_root, 'all_caption_tokenized_open_images.pkl'), 'rb') as f:
            prompts = pickle.load(f)
        print(prompts.shape)


        # sample_capid_inverse_idx = {}
        # for i, j in enumerate(sample_capid):
        #     sample_capid_inverse_idx[j] = i

        # =============================================================
        train_labels = torch.stack([torch.tensor(word_based_caption[i]) for i in word_based_caption])
        # print(train_labels.shape)
        # print(train_labels.sum())

        cls_dist_num_t = train_labels.sum(dim=0)
        print('class caption num:\n', cls_dist_num_t)
        count_list = torch.argsort(cls_dist_num_t)
        print(count_list)
        
        # reverse index: label idx 2 capid
        label_idx2capid = {}
        for i in range(cls_num):
            label_idx2capid[i] = []
        for i in tqdm(word_based_caption):
            cur_label = word_based_caption[i]
            for lidx in count_list:
                if cur_label[lidx] > 0.5:
                    label_idx2capid[lidx.item()].append(i)
                    break
        
        max_sample_per_cls = 1000
        train = []
        for Lidx in label_idx2capid:
            cur_cls_samples = min(max_sample_per_cls, len(label_idx2capid[Lidx]))
            for capid in label_idx2capid[Lidx][:cur_cls_samples]:
                i = capid - 1
                item_ = (prompts[i], word_based_caption[capid])
                train.append(item_)
        
        train_labels_ = torch.stack([torch.tensor(i[1]) for i in train])
        cls_dist_num_t_ = train_labels_.sum(dim=0)
        print('class caption num limited:\n', cls_dist_num_t_)
        print("===== Caption Distill Data: {} nums of word filtered caption  =====".format(len(word_based_caption)))
        

        # default template
        default_prompt_num = 10 # 1  100
        for i in range(cls_num):
            label = [0] * cls_num
            label[i] = 1
            tmp_p = clip.tokenize(prompt_template.format(object_categories[i]))[0]
            for j_ in range(default_prompt_num-1):
                train.append((tmp_p, torch.tensor(label)))
            
            for cur_temp in IMAGENET_TEMPLATES:
                tmp_p = clip.tokenize(cur_temp.format(object_categories[i]))[0]
                train.append((tmp_p, torch.tensor(label)))

        ############################
        ## test data
        path_labels = os.path.join(self.dataset_dir, 'TrainTestLabels')
        # labeled_data = dict()
        num_classes = len(object_categories)

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

        super().__init__(train_x=train, val=test[0::20], test=test, \
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
