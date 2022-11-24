import os
from os.path import join
from re import L
import pickle5 as pickle
import torch
import json
from tqdm import tqdm

from pycocotools.coco import COCO


prompt_template = "a photo of a {}."

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


####################################### VOC helpers #######################################
voc_object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
voc_classname_synonyms = [
    ['aeroplane', "air craft", "jet", "plane", "air plane"], 
    ['bicycle', 'bike', 'cycle'], 
    ['bird'], 
    ['boat', 'raft', 'dinghy'],
    ['bottle'], 
    ['bus', 'autobus', 'coach', 'charabanc', 'double decker', 'jitney', 'motor bus', 'motor coach', 'omnibus'], 
    ['car', 'taxi', 'auto', 'automobile', 'motor car'], 
    ['cat', 'kitty'], 
    ['chair', 'arm chair', 'bench'],
    ['cow'], 
    ['table', 'dining table', 'dinner table', 'din table'],  
    ['dog', 'pup', 'puppy', 'doggy'], 
    ['horse', 'colt', 'equus'],
    ['motor bike', 'motor cycle'], 
    ['person', 'human', 'people', 'man', 'woman', 'passenger'], 
    ['potted plant', 'house plant', 'bonsai', 'pot plant'],
    ['sheep'], 
    ['sofa', 'couch'], 
    ['train', 'rail way', 'railroad'], 
    ['tvmonitor', 'monitor', 'tv', 'television', 'telly']
]

def read_im_name_list(path):
    ret = []
    with open(path, 'r') as f:
        for line in f:
            tmp = line.strip().split(' ')
            ret.append(tmp[0])
    return ret

def read_image_label(file):
    print('[dataset] read ' + file)
    data_ = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.strip().split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data_[name] = label
    return data_

def read_object_labels(path, phase):
    path_labels = os.path.join(path, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(voc_object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, voc_object_categories[i] + '_' + phase + '.txt')
        data_ = read_image_label(file)

        if i == 0:
            for (name, label) in data_.items():
                labels = torch.zeros(num_classes).long()
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data_.items():
                labeled_data[name][i] = label
    return labeled_data



####################################### COCO helpers #######################################
coco_classname_synonyms = [
    ['person', 'human', 'people', 'man', 'woman', 'passenger'], 
    ['bicycle', 'bike', 'cycle'],
    ['car', 'taxi', 'auto', 'automobile', 'motor car'], 
    ['motor bike', 'motor cycle'], 
    ['aeroplane', "air craft", "jet", "plane", "air plane"], 
    ['bus', 'autobus', 'coach', 'charabanc', 'double decker', 'jitney', 'motor bus', 'motor coach', 'omnibus'],
    ['train', 'rail way', 'railroad'], 
    ['truck'],
    ['boat', 'raft', 'dinghy'],
    ['traffic light'],
    ['fire hydrant', 'fire tap', 'hydrant'],
    ['stop sign', 'halt sign'],
    ['parking meter'],
    ['bench'],
    ['bird'],
    ['cat', 'kitty'],
    ['dog', 'pup', 'puppy', 'doggy'],
    ['horse', 'colt', 'equus'],
    ['sheep'],
    ['cow'],
    ['elephant'],
    ['bear'],
    ['zebra'],
    ['giraffe', 'camelopard'],
    ['backpack', 'back pack', 'knapsack', 'packsack', 'rucksack', 'haversack'],
    ['umbrella'],
    ['handbag', 'hand bag', 'pocketbook', 'purse'],
    ['tie', 'necktie'],
    ['suitcase'],
    ['frisbee'],
    ['skis', 'ski'],
    ['snowboard'],
    ['sports ball', 'sport ball', 'ball', 'football', 'soccer', 'tennis', 'basketball', 'baseball'],
    ['kite'],
    ['baseball bat', 'baseball game bat'],
    ['baseball glove', 'baseball mitt', 'baseball game glove'],
    ['skateboard'],
    ['surfboard'],
    ['tennis racket'],
    ['bottle'],
    ['wine glass', 'vino glass'],
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
    ['donut', 'doughnut'],
    ['cake'],
    ['chair', 'arm chair'],
    ['couch', 'sofa'],
    ['potted plant', 'house plant', 'bonsai', 'pot plant'],
    ['bed'],
    ['dining table', 'dinner table', 'table', 'din table'], 
    ['toilet', 'commode'],
    ['tv', 'tvmonitor', 'monitor', 'television', 'telly'],
    ['laptop'],
    ['mouse'],
    ['remote'],
    ['keyboard'],
    ['cell phone', 'phone', 'mobile phone'],
    ['microwave'],
    ['oven', 'roaster'],
    ['toaster'],
    ['sink'],
    ['refrigerator', 'icebox'],
    ['book'],
    ['clock'],
    ['vase'],
    ['scissors'],
    ['teddy bear', 'teddy'],
    ['hair drier', 'blowing machine', 'hair dryer', 'dryer', 'blow dryer', 'blown dry', 'blow dry'],
    ['toothbrush'],
]

coco_object_categories = [syn[0] for syn in coco_classname_synonyms]



####################################### NUS helpers #######################################
nuswide_classname_synonyms =[
    ['airport', 'air port', 'air field', 'runway'],
    ['animal'],
    ['beach', 'plage', 'coast', 'seashore'],
    ['bear'],
    ['birds', 'bird'],
    ['boats', 'boat', 'raft', 'dinghy'],
    ['book'],
    ['bridge'],
    ['buildings', 'building'],
    ['cars', 'car'],
    ['castle'],
    ['cat', 'kitty'],
    ['cityscape', 'city', 'skyscraper'],
    ['clouds', 'cloud'],
    ['computer', 'desktop', 'laptop'],
    ['coral'],
    ['cow'],
    ['dancing', 'dance'],
    ['dog', 'pup', 'puppy', 'doggy'],
    ['earthquake', 'collapse building', 'break building', 'broken building'],
    ['elk', 'deer'],
    ['fire'],
    ['fish'],
    ['flags', 'flag'],
    ['flowers','flower'],
    ['food'],
    ['fox'],
    ['frost', 'forsted'],  # 'ice' 'frost'
    ['garden'],
    ['glacier', 'ice'], # 'iceberg'
    ['grass'],
    ['harbor', 'port', 'harbour'],
    ['horses', 'horse'],
    ['house'],
    ['lake'],
    ['leaf'],
    ['map'],
    ['military', 'army' , 'troops', 'troop'],
    ['moon'],
    ['mountain', 'hill'],
    ['nighttime', 'night time', 'night'],
    ['ocean', 'sea'],
    ['person', 'human', 'people', 'man', 'woman', 'passenger'],
    ['plane', 'aeroplane', "air craft", "jet", "air plane"],
    ['plants', 'plant'],
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
    ['swimmers', 'swimmer', 'swimming', 'swim'],
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
    ['wedding', 'engagement', 'bride', 'groom'],
    ['whales', 'whale'],
    ['window'],
    ['zebra'],
]

nuswide_object_categories = [syn[0] for syn in nuswide_classname_synonyms]


if __name__ == '__main__':
    for t in IMAGENET_TEMPLATES:
        print('\"' + t.format('[CLASS]') + '\"' )