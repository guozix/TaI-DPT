import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter

from tqdm import tqdm
import pickle5 as pickle

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .utils import soft_cross_entropy, softmax_sigmoid_BCEloss, \
    norm_logits_BCEloss, sigmoid_focal_loss, sigmoid_ASL_loss, ranking_loss, ASL_loss
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts, if_embedding=True, if_sequence=False):
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        
        if if_sequence:
            x = x @ self.text_projection  # NLD * Dd = NLd
            return x
        else:
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # ND * Dd = Nd
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
            return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.Caption.N_CTX
        ctx_init = cfg.TRAINER.Caption.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific double contexts")
                ctx_vectors_double = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_double, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized
        
        temperature = torch.tensor(3.0, dtype=dtype)  #  3.91 -> 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        # sigmoid_shift = torch.tensor(0.25, dtype=dtype)
        # self.sigmoid_shift = nn.Parameter(sigmoid_shift)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.Caption.CLASS_TOKEN_POSITION

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        ctx = self.ctx
        ctx_double = self.ctx_double
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if ctx_double.dim() == 2:
            ctx_double = ctx_double.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )


        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, prompts_neg, self.temperature, self.spatial_T, self.ranking_scale


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        # self.logit_scale = clip_model.logit_scale
        self.logit_scale = 100
        self.dtype = clip_model.dtype

        self.min_ = 5
        self.max_ = -5

    def forward(self, image=None, captions=None, if_test=False):
        if if_test:
            image_features = self.image_encoder(image.type(self.dtype))

            prompts, prompts_double, temperature, sigmoid_shift = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = 50 # temperature.exp() 
            # logit_scale = self.logit_scale  #.exp()
            logits = image_features @ text_features.t()
            # logits -= sigmoid_shift

            ## check score range
            # self.min_ = min(self.min_, logits.min().item())
            # self.max_ = max(self.max_, logits.max().item())
            # print(self.min_, self.max_)

            logits *= logit_scale

            return logits, image_features, text_features
        else:
            image_features = self.text_encoder(captions, None, if_embedding=False)
            # fake image feature, actually caption feature

            prompts, temperature, sigmoid_shift = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = 2.0 # temperature.exp() 
            # logit_scale = self.logit_scale  #.exp()
            logits = image_features @ text_features.t()
            # logits -= sigmoid_shift
            logits *= logit_scale

            # mean
            # max_ = torch.max(logits, dim=1, keepdim=True)[0]
            # min_ = torch.min(logits, dim=1, keepdim=True)[0]
            # mean_ = (max_ + min_) / 2
            # logits = logits - mean_

            return logits, image_features, text_features


class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        # self.image_encoder = clip_model.visual

        self.model = clip_model
        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.visual_encoder = IntermediateLayerGetter(self.model.visual, return_layers)
        self.positional_embedding = self.model.visual.attnpool.positional_embedding[1::]
        self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.model.visual.attnpool.c_proj.bias
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
    
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
    
    def forward(self, image=None, captions=None, if_test=False):
        if if_test:        
            image_feat = self.encode_image(image)
            b, c, h, w = image_feat.shape
            x = image_feat.reshape(b, c, h * w).permute(2, 0, 1)
            # g_x = x.mean(0, keepdim=True)
            # x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW)xBxC        
            
            x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
            x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
            image_features = x

            # image_feature_ = self.model.visual(image.type(self.dtype))
            image_feature_, _ = self.model.visual.attnpool(image_feat, if_pos=False)
            # ===============================================================

            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)
            
            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50
            logits_ = logit_scale * image_feature_ @ text_features.t()   # B * C,  cls * C, = B * cls
            logits_neg = image_features @ text_features_neg.t()    #  HW * B * C,  cls * C,  HW * B * cls
           
            tmp_scale = spatial_T.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_image  # 5 #
            prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale, dim=0)
            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial, dim=0)

            # logits = logits.max(dim=0)[0]
            # logits = (logits - logits_neg)
            # logits = ((logits - logits_neg) + (logits_ - logits_neg_))/2
            return logits_, logits_local, logits_neg, image_features @ text_features.t()  # compare additional branch with global proxy
        else:
            image_feat = self.text_encoder(captions, None, if_embedding=False, if_sequence=True) 
            # b, l, d = image_feat.shape
            image_feature_ = image_feat[torch.arange(image_feat.shape[0]), captions.argmax(dim=-1)]  # BD
            image_features = image_feat.permute(1, 0, 2)  # LBD
            # ===============================================================

            prompts, prompts_double, temperature, spatial_T, rk_scale = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_neg = self.text_encoder(prompts_double, tokenized_prompts)

            image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
            
            # mask irrelavent tokens
            text_mask = (captions == 0).long() * (-10000)  # BL

            logit_scale = temperature.exp()  # rk_scale
            logit_scale = logit_scale if self.cfg.TRAIN.IF_LEARN_SCALE else 4.0 # 50 # temperature.exp()  # self.logit_scale.exp()
            logits_ = logit_scale * image_feature_ @ text_features.t()   # B * C,  cls * C, = B * cls
            logits_neg = image_features @ text_features_neg.t()    #  L * B * C,  cls * C,  L * B * cls
            logits_neg = logits_neg.permute(2, 1, 0) + text_mask[None, :, :]
            logits_neg = logits_neg.permute(2, 1, 0)

            tmp_scale = spatial_T.exp() if self.cfg.TRAIN.IF_LEARN_spatial_SCALE else self.cfg.TRAIN.spatial_SCALE_text
            prob_spatial = torch.nn.functional.softmax(logits_neg * tmp_scale, dim=0)
            logits_local = torch.sum(logit_scale * logits_neg * prob_spatial, dim=0)
            
            # logits = logits.max(dim=0)[0]
            # logits = (logits - logits_neg)
            # logits = ((logits - logits_neg) + (logits_ - logits_neg_))/2
            return logits_, logits_local, image_features, text_features


# kl_loss = nn.KLDivLoss(reduction="batchmean")
# ce_loss = torch.nn.CrossEntropyLoss()

@TRAINER_REGISTRY.register()
class Caption_distill_double(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def model_inference(self, input):
        return self.model(input, if_test=True)
        # return self.model(None, input)

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
        images = []
        labels = []
        outputs = []
        output_poss = []
        spatial_logits = []
        global_spatial_logits = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            # output = self.model_inference(input)
            output, output_pos, image_features_, text_features_ = self.model_inference(input)
            self.evaluator.process(output, label, output_pos)
            # images.append(input.clone().detach().cpu())
            # labels.append(label.clone().detach().cpu().numpy())
            # outputs.append(output.clone().detach().cpu().numpy())
            # output_poss.append(output_pos.clone().detach().cpu().numpy())
            # spatial_logits.append(image_features_.clone().detach().cpu().numpy())
            # global_spatial_logits.append(text_features_.clone().detach().cpu().numpy())

        # with open('vis_VOC2007_testset_input_result.pkl', 'wb') as f:
        #     pickle.dump((images, labels, outputs, output_poss, spatial_logits, global_spatial_logits), f)
            
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

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Caption.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        print('==================== Building model in Caption_distill_double ======================')
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model = DenseCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
                                                            
        # load_pretrained_weights(self.model.prompt_learner, 'output/voc2007_caption_distill_abinf/Caption_distill_double/rn50_fixscale/nctx16_cscFalse_ctpend/seed3/prompt_learner/model-best.pth.tar')
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.Caption.PREC
        if prec == "amp":
            with autocast():
                output, output_local, _, _ = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local, _, _ = self.model(None, image)
            # loss = F.cross_entropy(output, label)
            # loss = kl_loss(output, label)
            if   self.cfg.TRAIN.LOSSFUNC == 'sigmoid':
                loss = norm_logits_BCEloss(output, label.float()) + norm_logits_BCEloss(output_local, label.float())
            elif self.cfg.TRAIN.LOSSFUNC == 'focal':
                loss = sigmoid_focal_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'asl':
                loss = ASL_loss(output, label) + ASL_loss(output_local, label.float())
            elif self.cfg.TRAIN.LOSSFUNC == 'ranking':
                loss = ranking_loss(output, label)
            elif self.cfg.TRAIN.LOSSFUNC == 'double_ranking':
                loss = ranking_loss(output, label, scale_ = 1.0, margin_ = 1) + ranking_loss(output_local, label, scale_ = 1.0, margin_ = 1)
            elif self.cfg.TRAIN.LOSSFUNC == 'normlabel_softmax':
                label = label / label.sum(dim=-1, keepdim=True)
                loss = soft_cross_entropy(output, label)
            else:
                loss = soft_cross_entropy(output, label)

            # loss = norm_logits_BCEloss(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            print(state_dict)
