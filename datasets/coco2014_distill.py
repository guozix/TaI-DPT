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
from trainers.coop import load_clip_to_cpu

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing
from pycocotools.coco import COCO

from .data_helpers import *

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

object_categories = coco_object_categories
classname_synonyms = coco_classname_synonyms

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
class COCO2014_distill(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'COCO'
        cls_num = len(object_categories)
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        coco_instance_json_file = os.path.join(self.dataset_dir, "annotations/instances_val2014.json")

        coco = COCO(coco_instance_json_file)
        self.valset_ids = coco.getImgIds()
        

        instance_info = {}
        with open(coco_instance_json_file, 'r') as f:
            instance_info = json.load(f)

        clsid2clsidx = {}
        clsidx2clsid = {}
        clsid2clsname = {}
        for idx, cat_info in enumerate(instance_info["categories"]):
            clsid2clsidx[cat_info['id']] = idx
            clsidx2clsid[idx] = cat_info['id']
            clsid2clsname[cat_info['id']] = cat_info['name']

        test_imgdir = [self.dataset_dir + '/val2014/{}'.format(coco.loadImgs(ids = imgid)[0]['file_name']) for imgid in self.valset_ids]
        test_label = torch.zeros((len(self.valset_ids), cls_num), dtype=torch.long)
        for idx, imgid in enumerate(self.valset_ids):
            annIds = coco.getAnnIds(imgIds = imgid)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                tmp_idx = clsid2clsidx[ann['category_id']]
                test_label[idx, tmp_idx] = 1

        test = []
        for i in range(len(self.valset_ids)):
            item_ = Datum(impath=test_imgdir[i], label=test_label[i], classname='')
            test.append(item_)


        # ===================  training captions
        caption_feat_root = os.getcwd()
        with open(join(caption_feat_root, 'coco_caption_text_embed_sampled_idx.pkl'), 'rb') as f:
            sample_capid = pickle.load(f)

        # if exists, just read label
        if os.path.exists(join(caption_feat_root, 'coco2014_cls_word_based_caption_labels.pkl')):
            with open(join(caption_feat_root, 'coco2014_cls_word_based_caption_labels.pkl'), 'rb') as f:
                word_based_caption = pickle.load(f)
        
        else:
            coco_root = self.dataset_dir
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

            word_based_caption = {} # capid 2 cls labels
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
            with open('coco2014_cls_word_based_caption_labels.pkl', 'wb') as f:
                pickle.dump(word_based_caption, f)
            with open('coco2014_cls_capid_filterword_empty.pkl', 'wb') as f:
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
        train = []
        for capid in word_based_caption:
            i = sample_capid_inverse_idx[capid]
            item_ = (prompts[i], torch.tensor(word_based_caption[capid]))
            train.append(item_)
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

        super().__init__(train_x=train, val=test[0::10], test=test, \
            num_classes=len(object_categories), classnames=object_categories, \
            lab2cname={idx: classname for idx, classname in enumerate(object_categories)})

    def read_name_list(self, path):
        ret = []
        with open(path, 'r') as f:
            for line in f:
                tmp = line.strip().split(' ')
                ret.append(tmp[0])
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
