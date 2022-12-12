
# Texts as Images in Prompt Tuning for Multi-Label Recognition


## Introduction

This repo is the official implementation of **Texts as Images in Prompt Tuning for Multi-Label Recognition**.

[[arxiv](https://arxiv.org/abs/2211.12739)]

TaI-DPT explores the feasibility of prompting with text data for multi-label image recognition. Notable improvements are observed compared to zero-shot methods on multiple common multi-label benchmarks. For more details please see the [paper](https://arxiv.org/abs/2211.12739).

Contact us with zixian_guo@foxmail.com


<center>
<img src="./figures/cvpr2023figbig.png">

Fig.1 Overview of Text-as-Image (TaI) prompting.
</center>

## Install

The code is based largely on the implementation of [CoOp](https://github.com/KaiyangZhou/CoOp) and [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch).


Please follow the steps below to build your environment.

```bash
# Create a conda environment (Omit if you already have a suitable environment)
conda create -n dassl python=3.7
conda activate dassl
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge # torch (version >= 1.7.1)

# Clone this repo
git clone https://github.com/guozix/TaI-DPT.git
cd TaI-DPT

# install Dassl
cd Dassl.pytorch-master/
# Install dependencies
pip install -r requirements.txt
# Install this library (no need to re-build if the source code is modified)
python setup.py develop

cd ..
# Install CLIP dependencies
pip install -r requirements.txt

# Finished
```

## Datasets
We use captions from MS-COCO and localized narratives from OpenImages, and we evaluate our method on VOC2007, MS-COCO and NUS-WIDE.
The directory structure is organized as follows.
```
DATA
├── OpenImages
│   ├── captions
│   │   └── open_images_train_v6_captions.jsonl
├── VOCdevkit
│   ├── VOC2007
|   │   ├── Annotations
|   │   ├── caption_data
|   │   ├── ImageSets
|   │   │   ├── Layout
|   │   │   ├── Main
|   │   │   └── Segmentation
|   │   ├── JPEGImages
|   │   ├── SegmentationClass
|   │   └── SegmentationObject
├── COCO
│   ├── annotations
│   ├── train2014
│   └── val2014
└── NUSWIDE
    ├── ImageList
    │   ├── Imagelist.txt
    │   ├── TestImagelist.txt
    │   └── TrainImagelist.txt
    ├── Flickr
    │   ├── actor
    │   ├── administrative_assistant
    │   ├── adobehouses
    │   ├── adult
    │   ...
    ├── TrainTestLabels
    └── Concepts81.txt
```
<!-- We provide images of NUS-WIDE used in our experiments:
https://pan.baidu.com/s/1Bj-7fdrZAvUJPqAKrUkbbQ  (verification code: s6oj) -->

## Usage
**Reproduce of Table 1.**

Test Baseline ZSCLIP on the datasets:
``` bash
cd scripts/
bash zsclip.sh voc2007_distill rn50
bash zsclip.sh coco2014_distill rn50
bash zsclip.sh nuswide_distill_limit rn50
```

Train TaI-DPT on the datasets:
``` bash
cd scripts/
bash main.sh voc2007_distill rn50_voc2007 end 16 False voc2007_caption
bash main.sh coco2014_distill rn50_coco2014 end 16 False coco2014_caption
bash main.sh nuswide_distill_limit rn50_nuswide end 16 False nuswide_caption
```

**Reproduce of DualCoOp**
``` bash
cd scripts/

```


## Thanks

We use code from [CoOp](https://github.com/KaiyangZhou/CoOp) and [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), which are great repositories and we encourage you to check them out and cite them in your work.