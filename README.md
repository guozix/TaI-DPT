
# Texts as Images in Prompt Tuning for Multi-Label Recognition


## Introduction

This repo officially implements **Texts as Images in Prompt Tuning for Multi-Label Recognition**.

[[arxiv](https://arxiv.org/abs/2211.12739)] [[CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Texts_as_Images_in_Prompt_Tuning_for_Multi-Label_Image_Recognition_CVPR_2023_paper.html)]

TaI-DPT explores the feasibility of prompting with text data for multi-label image recognition. Notable improvements are observed compared to zero-shot methods on multiple common multi-label benchmarks. For more details, please see the [paper](https://arxiv.org/abs/2211.12739).

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

The multi-label classification datasets can be accessed from their official websites [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/), [MSCOCO2014](https://cocodataset.org/#download) and [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html). The raw images of NUS-WIDE can be accessed from [here](https://pan.baidu.com/s/1Bj-7fdrZAvUJPqAKrUkbbQ) (verification code: s6oj).

For the OpenImages dataset, we only use the localized narratives of its "V6" version. The ~130MB jsonl file can be downloaded from the official [site](https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl).

```
DATAROOT
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
**To evaluate the results in Table 1.**

Change the `DATA` variable in the bash scripts to the `DATAROOT` path above.

Test Baseline ZSCLIP on the datasets:
``` bash
cd scripts/
bash zsclip.sh voc2007_distill rn50
bash zsclip.sh coco2014_distill rn50
bash zsclip.sh nuswide_distill_limit rn50
```

The `mAP score default` term in the output denotes the mAP score calculated by using only the global classification logits, i.e. baseline CLIP.
The `mAP score merged` term denotes the performance of equally merging the logits from the glocal and the local branch, i.e. CLIP-DPT, similarly hereinafter.

Evaluate TaI-DPT:

Unzip the trained model weights in output.zip.
The file directory should be like:

```
output
├── coco2014_caption
│   └── Caption_distill_double
│       └── rn50_coco2014
│           └── nctx16_cscFalse_ctpend
│               ├── seed1
│                   ...
│               ├── seed2
│                   ...
│               └── seed3
│                   ...
├── nuswide_caption
│   └── Caption_distill_double
│       └── rn50_nuswide
│           └── nctx16_cscFalse_ctpend
│               ├── seed1
│                   ...
│               ├── seed2
│                   ...
│               └── seed3
│                   ...
└── voc2007_caption
    └── Caption_distill_double
        └── rn50_voc2007
            └── nctx16_cscFalse_ctpend
                ├── seed1
                    ...
                ├── seed2
                    ...
                └── seed3
                    ...
```

Then run the evaluation script:
``` bash
cd scripts/
bash main_eval.sh voc2007_distill rn50_voc2007 end 16 False voc2007_caption
bash main_eval.sh coco2014_distill rn50_coco2014 end 16 False coco2014_caption
bash main_eval.sh nuswide_distill_limit rn50_nuswide end 16 False nuswide_caption
```

The `mAP score default` and  `mAP score merged` term in the output denotes mAP score of TaI and TaI-DPT. The results are also saved in log files in `./output/evaluation`. 


**To reproduce the results in Table 1.**

(If you extracted the trained model in the previous evaluation step, you need to set a different `run_ID`, otherwise the script will skip the training.)

Train TaI-DPT on the datasets:
``` bash
cd scripts/
bash main.sh voc2007_distill rn50_voc2007 end 16 False voc2007_caption
bash main.sh coco2014_distill rn50_coco2014 end 16 False coco2014_caption
bash main.sh nuswide_distill_limit rn50_nuswide end 16 False nuswide_caption
```

**To reproduce DualCoOp**
``` bash
cd scripts/
# VOC2007
bash main_dual.sh voc2007_partial rn101 end 16 True voc2007_partial_dualcoop_448_CSC_p0_1 0.1 0
bash main_dual.sh voc2007_partial rn101 end 16 True voc2007_partial_dualcoop_448_CSC_p0_2 0.2 0
bash main_dual.sh voc2007_partial rn101 end 16 True voc2007_partial_dualcoop_448_CSC_p0_3 0.3 0
...

# COCO2014
bash main_dual.sh coco2014_partial rn101 end 16 True coco2014_partial_dualcoop_448_CSC_p0_1 0.1 1
...

# NUS-WIDE
bash main_dual.sh nuswide_partial rn101_nus end 16 True nuswide_partial_dualcoop_448_CSC_p0_1 0.1 2
...

```

**To reproduce ensemble results**

The parameter setting here is a little cumbersome, but the logic is simple. You only need to specify two trained model paths and the corresponding model configurations.

Here is an example of ensemble of TaI-DPT and DualCoOp on VOC2007. You should finish training TaI-DPT and DualCoOp (under a specific `partial_prob`) on VOC2007 before.

``` bash
cd scripts/

bash ensemble.sh voc2007_distill rn50_voc2007 end 16 False voc2007_caption_e tmp1.pkl \
 output/voc2007_caption/Caption_distill_double/rn50_voc2007/nctx16_cscFalse_ctpend/seed1 \
voc2007_partial rn101 end 16 True voc2007_partial_dualcoop_448_CSC_p0_5_e tmp2.pkl \
 output/voc2007_partial_dualcoop_448_CSC_p0_5/Caption_dual/rn101/nctx16_cscTrue_ctpend/seed1 0.9
```
## Thanks

We use code from [CoOp](https://github.com/KaiyangZhou/CoOp) and [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), which are great repositories and we encourage you to check them out and cite them in your work.
