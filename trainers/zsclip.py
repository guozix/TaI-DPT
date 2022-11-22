import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from tqdm import tqdm

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",

    # "VOC2012_trainset": "a photo of a {}.",
    # "VOC2012_trainset_gt": "a photo of a {}.",
    # "VOC2012_testset": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        try:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        except:
            print('!! WARNING: Not found template for {}'.format(cfg.DATASET.NAME))
            temp = "a photo of a {}."

        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

        self.min_ = 100000
        self.max_ = -100000

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        # logits = image_features @ self.text_features.t()

        ## check score range
        # print('logits scale: ', logit_scale)
        # self.min_ = min(self.min_, logits.min().item())
        # self.max_ = max(self.max_, logits.max().item())
        # print(self.min_, self.max_)

        return logits, None, None


@TRAINER_REGISTRY.register()
class ZeroshotCLIP_dense(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        
        try:
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        except:
            print('!! WARNING: Not found template for {}'.format(cfg.DATASET.NAME))
            temp = "a photo without {}."  # temp = "a photo of a {}."  # 

        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

        self.visual_encoder = IntermediateLayerGetter(self.clip_model.visual, {"layer4": "0"})
        self.positional_embedding = self.clip_model.visual.attnpool.positional_embedding[1::]
        self.v_linear_weight = self.clip_model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.clip_model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.clip_model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.clip_model.visual.attnpool.c_proj.bias

        self.min_ = 1
        self.max_ = 0

    def encode_image(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x

        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)
        return x

    def model_inference(self, image):
        image_feat = self.encode_image(image)
        b, c, h, w = image_feat.shape
        x = image_feat.reshape(b, c, h * w).permute(2, 0, 1)

        x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
        x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
        image_features = x
        
        image_feature_, _ = self.clip_model.visual.attnpool(image_feat)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp() # logit_scale = self.clip_model.logit_scale.exp()
        # logits_tmp = logit_scale * image_features @ self.text_features.t()   #  HW * B * C,  cls * C,  HW * B * cls
        logits_ = logit_scale * image_feature_ @ self.text_features.t()   # B * C,  cls * C, = B * cls
        logits = logit_scale * image_features @ self.text_features.t()    #  HW * B * C,  cls * C,  HW * B * cls
 
        
        # logits = torch.max(logits_tmp, dim=0)[0]
        # logits_tmp_ = F.softmax(logits_tmp * 50, dim=0)
        # logits = torch.sum((logits_tmp * logits_tmp_), dim=0)

        # ## check score range
        # self.min_ = min(self.min_, logits.min().item())
        # self.max_ = max(self.max_, logits.max().item())
        # print(self.min_, self.max_)
        
        prob_spatial = torch.nn.functional.softmax(logits, dim=0)
        logits = torch.sum(logits * prob_spatial, dim=0)
        # logits = logits.max(dim=0)[0]


        return logits, logits_, None, None

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        # save_feature_forvis = []
        # save_text_feature = []
        # save_label = []
        # save_logits = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # output = self.model_inference(input)
            output, output_pos, image_features_, text_features_ = self.model_inference(input)
            self.evaluator.process(output, label, output_pos)
            

        #     save_text_feature.append(text_features_.clone().detach().cpu().numpy())
        #     save_feature_forvis.append(image_features_.clone().detach().cpu().numpy())
        #     save_label.append(label.clone().detach().cpu().numpy())
        #     save_logits.append(output.clone().detach().cpu().numpy())
        # save_feature_forvis = np.concatenate(save_feature_forvis, axis=0)
        # save_text_feature = np.stack(save_text_feature, axis=0)
        # save_label = np.concatenate(save_label, axis=0)
        # save_logits = np.concatenate(save_logits, axis=0)
        # np.savez("test_features.npz", image_features=save_feature_forvis, text_features=save_text_feature, labels=save_label)
        # np.savez("logits_labels.npz", logits=save_logits, labels=save_label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
