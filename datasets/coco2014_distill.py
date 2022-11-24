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

from .oxford_pets import OxfordPets

prompt_template = "a photo of {}."


IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

classname_synonyms =[
    ['person', 'human', 'people', 'man', 'woman', 'passenger'],
    ['bicycle', 'bike', 'cycle'],
    ['car', 'taxi'],
    ['motorcycle', 'motorbike'],
    ['airplane', 'aeroplane', 'aircraft', 'jet', 'plane',],
    ['bus'],
    ['train', 'railway'],
    ['truck'],
    ['boat'],
    ['traffic light'],
    ['fire hydrant'],
    ['stop sign'],
    ['parking meter'],
    ['bench'],
    ['bird'],
    ['cat', 'kitty'],
    ['dog', 'pup', 'puppy', 'doggy'],
    ['horse', 'colt'],
    ['sheep'],
    ['cow'],
    ['elephant'],
    ['bear'],
    ['zebra'],
    ['giraffe'],
    ['backpack'],
    ['umbrella'],
    ['handbag'],
    ['tie'],
    ['suitcase'],
    ['frisbee'],
    ['skis'],
    ['snowboard'],
    ['sports ball'],
    ['kite'],
    ['baseball bat'],
    ['baseball glove'],
    ['skateboard'],
    ['surfboard'],
    ['tennis racket'],
    ['bottle'],
    ['wine glass'],
    ['cup'],
    ['fork'],
    ['knife'],
    ['spoon'],
    ['bowl'],
    ['banana'],
    ['apple'],
    ['sandwich'],
    ['orange'],
    ['broccoli'],
    ['carrot'],
    ['hot dog'],
    ['pizza'],
    ['donut'],
    ['cake'],
    ['chair', 'armchair', 'bench'],
    ['couch', 'sofa'],
    ['potted plant', 'pottedplant', 'houseplants', 'bonsai'],
    ['bed'],
    ['dining table', 'diningtable', 'dinnertable', 'table'],
    ['toilet'],
    ['tv', 'tvmonitor', 'monitor', 'television'],
    ['laptop'],
    ['mouse'],
    ['remote'],
    ['keyboard'],
    ['cell phone', 'phone', 'mobile phone'],
    ['microwave'],
    ['oven'],
    ['toaster'],
    ['sink'],
    ['refrigerator'],
    ['book'],
    ['clock'],
    ['vase'],
    ['scissors'],
    ['teddy bear'],
    ['hair drier'],
    ['toothbrush'],
]


clsname2idx = {}
for idx, synset in enumerate(classname_synonyms):
    for n in synset:
        clsname2idx[n] = idx
        # clsname2idx[n+'s'] = idx
        # clsname2idx[n+'es'] = idx

        if ' ' in n:
            m = n.replace(' ', '')
            clsname2idx[m] = idx
            # clsname2idx[m+'s'] = idx
            # clsname2idx[m+'es'] = idx

object_categories = [syn[0] for syn in classname_synonyms]


@DATASET_REGISTRY.register()
class COCO2014_distill(DatasetBase):
    def __init__(self, cfg):
        self.dataset_dir = 'COCO'
    
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        coco_instance_json_file = os.path.join(self.dataset_dir, "annotations2014/instances_val2014.json")

        coco = COCO(coco_instance_json_file)
        self.valset_ids = coco.getImgIds()
        cls_num = 80

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
        caption_feat_root = '/home/qiangwenjie/gzx/CoOp-main'  # '/home/weiyuxiang/gzx/CoOp-main'
        with open(join(caption_feat_root, 'coco_caption_text_embed_sampled_idx.pkl'), 'rb') as f:
            sample_capid = pickle.load(f)

        # ##### if exists, just read label
        if os.path.exists(join(caption_feat_root, 'coco2014_cls_word_based_caption_labels_v3.pkl')):

            with open(join(caption_feat_root, 'coco2014_cls_word_based_caption_labels_v4.pkl'), 'rb') as f:
                word_based_caption = pickle.load(f)
            with open(join(caption_feat_root, 'coco2014_cls_capid_filterword_empty_v4.pkl'), 'rb') as f:
                capid_empty_filter = pickle.load(f)
            with open(join(caption_feat_root, 'coco2014_cls_feature_based_caption_selected_1w_percls_v1.pkl'), 'rb') as f:
                feature_based_caption = pickle.load(f)
        
        else:
            ## not exists, make now
            with open(join(caption_feat_root, 'coco_caption_text_embed_sampled.pkl'),'rb') as f:
                sample_text_embed = pickle.load(f)
            print('caption embedding shape:', sample_text_embed.shape)  # torch.Size([118287, 1024])

            with open(join(caption_feat_root, "coco_cls_prompts_embed.pkl"), "rb") as f:
                coco_default_cls_embed = pickle.load(f)

            coco_root = self.dataset_dir # os.path.join(root, "COCO")  # "/home/weiyuxiang/gzx/VOS/datas/COCO"
            coco_caption_json_file = os.path.join(coco_root, "annotations/captions_train2017.json")
            caption_info = {}
            with open(coco_caption_json_file, 'r') as f:
                caption_info = json.load(f)

            anno_id2path = {}
            for i in caption_info["annotations"]:
                anno_id2path[i["id"]] = i
            # print(i.keys())
            print("captions_train2017 nums:", len(anno_id2path))

            sample_text_embed = sample_text_embed.cuda()
            coco_default_cls_embed = coco_default_cls_embed.cuda()
            logits_all = ( coco_default_cls_embed.cuda() @ sample_text_embed.t() ).float().cpu() 
            del coco_default_cls_embed
            del sample_text_embed

            print("corr matrix shape:", logits_all.shape)  # torch.Size([20, 118287])
            print('Ranking...')
            sorted_idxs = torch.argsort(logits_all, dim=1, descending=True)

            base_caption_nums = 10000
            feature_based_caption = {}  # clsidx 2 capid
            for i in range(cls_num):
                for idx in sorted_idxs[i, :base_caption_nums]:
                    feature_based_caption.setdefault(i, list())
                    feature_based_caption[i].append(sample_capid[idx])
            with open('coco2014_cls_feature_based_caption_selected_5kpercls.pkl', 'wb') as f:
                pickle.dump(feature_based_caption, f)
                # 80 keys denote class id, 10000 len list corresponded, denote similar caption id.

            word_based_caption = {} # capid 2 cls labels
            capid_empty_filter = set()
            for i, capid in enumerate(tqdm(sample_capid)):
                cap = anno_id2path[capid]['caption'].lower()
                L = [0] * 80
                flag = 0
                for name in clsname2idx:
                    if name in cap:
                        L[clsname2idx[name]] = 1
                        flag = 1
                if flag:
                    word_based_caption[capid] = L
                else:
                    capid_empty_filter.add(capid)

            print('===== Filtered by words all captions, num of captions contains object {}, num of caption empty {} ====='.format(len(word_based_caption), len(capid_empty_filter)))
            with open('coco2014_cls_word_based_caption_labels.pkl', 'wb') as f:
                pickle.dump(word_based_caption, f)
                # len=118287, caption id to label list
            with open('coco2014_cls_capid_filterword_empty.pkl', 'wb') as f:
                pickle.dump(capid_empty_filter, f)

        if os.path.exists(join(caption_feat_root, 'all_caption_tokenized.pkl')):
            with open(join(caption_feat_root, 'all_caption_tokenized.pkl'), 'rb') as f:
                prompts = pickle.load(f)
        # else:
        #     prompts = torch.cat([clip.tokenize(anno_id2path[p]['caption']) for p in sample_capid])
        #     with open('all_caption_tokenized.pkl', 'wb') as f:
        #         pickle.dump(prompts, f)

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
