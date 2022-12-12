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
import jsonlines
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

object_categories = nuswide_object_categories
classname_synonyms = nuswide_classname_synonyms

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


@DATASET_REGISTRY.register()
class nuswide_distill_limit(DatasetBase):
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

        caption_feat_root = os.getcwd()
        
        # if exists, just read label
        if os.path.exists(join(caption_feat_root, 'nuswide_cls_word_based_caption_labels.pkl')):
            with open(join(caption_feat_root, 'nuswide_cls_word_based_caption_labels.pkl'), 'rb') as f:
                word_based_caption = pickle.load(f)
        else:
            def get_wordnet_pos(tag):
                if tag.startswith('J'):
                    return wordnet.ADJ
                elif tag.startswith('V'):
                    return wordnet.VERB
                elif tag.startswith('N'):
                    return wordnet.NOUN
                elif tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return None

            wnl = WordNetLemmatizer()

            cnt = 0
            tmp = []
            word_based_caption = {} # capid 2 cls labels
            capid_empty_filter = set()
            with open(join(root, 'OpenImages/captions/open_images_train_v6_captions.jsonl'), 'r+', encoding='utf-8') as f:
                print("Start parsing captions from openimages ...")
                for item in tqdm(jsonlines.Reader(f)):
                    # tmp.append(clip.tokenize(item['caption'], truncate=True))
                    
                    cnt += 1
                    i = cnt
                    # if cnt % 1000 == 0:
                    #     print(cnt)

                    cap = item['caption'].lower()
                    
                    noum_list = word_tokenize(cap)[:77]  # clip only encoder sentence shorter than 77
                    tagged_sent = pos_tag(noum_list) 
                    # print(tagged_sent)
                    # break

                    lemmas_sent = []
                    for tag in tagged_sent:
                        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) 
                    # print(lemmas_sent)

                    cap = ' ' + ' '.join(lemmas_sent) + ' '

                    L = [0] * 81
                    flag = 0
                    for name in nameset_compound:
                        name_ = ' ' + name + ' '
                        if (name_ in cap):
                            L[clsname2idx_[name]] = 1
                            flag = 1
                            cap = cap.replace(name_, ' ')
                    for name in nameset:
                        name_ = ' ' + name + ' '
                        if (name_ in cap):
                            L[clsname2idx_[name]] = 1
                            flag = 1
                            cap = cap.replace(name_, ' ')
                    
                    if flag:
                        word_based_caption[i] = L
                    else:
                        capid_empty_filter.add(i)
                print('===== Filtered by words all captions, num of captions contains object {}, num of caption empty {} ====='.format(len(word_based_caption), len(capid_empty_filter)))
                with open(f'nuswide_cls_word_based_caption_labels.pkl', 'wb') as f:
                    pickle.dump(word_based_caption, f)
                with open(f'nuswide_cls_capid_filterword_empty.pkl', 'wb') as f:
                    pickle.dump(capid_empty_filter, f)

        if os.path.exists(join(caption_feat_root, 'all_caption_tokenized_open_images.pkl')):
            with open(join(caption_feat_root, 'all_caption_tokenized_open_images.pkl'), 'rb') as f:
                prompts = pickle.load(f)
        else:
            tmp = []
            with open(join(root, 'OpenImages/captions/open_images_train_v6_captions.jsonl'), 'r+', encoding='utf-8') as f:
                for item in tqdm(jsonlines.Reader(f), desc='tokenizing openimage captions ...'):
                    tmp.append(clip.tokenize(item['caption'], truncate=True))
                    # cnt += 1
                    # if cnt % 100 == 0:
                    #     print(cnt)

            prompts = torch.cat(tmp)
            with open('all_caption_tokenized_open_images.pkl', 'wb') as f:
                pickle.dump(prompts, f)
        print(prompts.shape)
        
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
        default_prompt_num = 10
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
