from calendar import c
import os
from os.path import join
from re import L
import pickle5 as pickle
import random
from scipy.io import loadmat
from collections import defaultdict
import torch
import json
from tqdm import tqdm
from clip import clip
from clip.model import convert_weights

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

object_categories = voc_object_categories
classname_synonyms = voc_classname_synonyms

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
class VOC2007_distill(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'VOCdevkit/VOC2007'
        cls_num = len(object_categories)
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "JPEGImages")

        caption_feat_root = os.getcwd()

        with open(join(caption_feat_root, 'coco_caption_text_embed_sampled_idx.pkl'), 'rb') as f:
            sample_capid = pickle.load(f)
        # if exists, just read label
        if os.path.exists(join(caption_feat_root, 'voc2007_cls_word_based_caption_labels.pkl')):
            with open(join(caption_feat_root, 'voc2007_cls_word_based_caption_labels.pkl'), 'rb') as f:
                word_based_caption = pickle.load(f)

        else:
            coco_root = os.path.join(root, "COCO")
            coco_caption_json_file = os.path.join(coco_root, "annotations/captions_train2017.json")
            caption_info = {}
            with open(coco_caption_json_file, 'r') as f:
                caption_info = json.load(f)

            anno_id2path = {}
            for i in caption_info["annotations"]:
                anno_id2path[i["id"]] = i
            # print(i.keys())
            print("captions_train2017 nums:", len(anno_id2path))

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

            word_based_caption = {} # capid 2 cls_labels
            capid_empty_filter = set()
            wnl = WordNetLemmatizer()
            for i, capid in enumerate(tqdm(sample_capid)):
                cap = anno_id2path[capid]['caption'].lower()
                noum_list = word_tokenize(cap)
                tagged_sent = pos_tag(noum_list) 
                # print(tagged_sent)
                # break

                lemmas_sent = []
                for tag in tagged_sent:
                    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                    lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
                # print(lemmas_sent)

                cap = ' ' + ' '.join(lemmas_sent) + ' '

                L = [0] * cls_num
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
                    word_based_caption[capid] = L
                else:
                    capid_empty_filter.add(capid)
            print('===== Filtered by words all captions, num of captions contains object {}, num of caption empty {} ====='.format(len(word_based_caption), len(capid_empty_filter)))
            with open(f'voc2007_cls_word_based_caption_labels.pkl', 'wb') as f:
                pickle.dump(word_based_caption, f)
            with open(f'voc2007_cls_capid_filterword_empty.pkl', 'wb') as f:
                pickle.dump(capid_empty_filter, f)

        if os.path.exists(join(caption_feat_root, 'all_caption_tokenized.pkl')):
            with open(join(caption_feat_root, 'all_caption_tokenized.pkl'), 'rb') as f:
                prompts = pickle.load(f)
        else:
            prompts = torch.cat([clip.tokenize(anno_id2path[p]['caption']) for p in sample_capid])
            with open('all_caption_tokenized.pkl', 'wb') as f:
                pickle.dump(prompts, f)

        sample_capid_inverse_idx = {}
        for i, j in enumerate(sample_capid):
            sample_capid_inverse_idx[j] = i

        # =============================================================
        if not cfg.TRAIN.IF_ablation:
            caption_set = word_based_caption.keys()
        else:
            cap_ids = list(word_based_caption.keys())
            caption_set = random.sample(cap_ids, min(cfg.TRAIN.Caption_num, len(cap_ids)))
        
        train = []
        for capid in caption_set:
            i = sample_capid_inverse_idx[capid]
            item_ = (prompts[i], torch.tensor(word_based_caption[capid]))
            train.append(item_)
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

        ############################################################################################################
        ## test data
        test_data_imname2label = read_object_labels(self.dataset_dir, phase='test')
        self.im_name_list_test = read_im_name_list(join(self.dataset_dir, 'ImageSets/Main/test.txt'))
    
        test = []
        for name in self.im_name_list_test:
            item_ = Datum(impath=self.image_dir+'/{}.jpg'.format(name), label=test_data_imname2label[name], classname='')
            test.append(item_)

        super().__init__(train_x=train, val=test, test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})
