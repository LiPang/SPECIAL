import numpy as np
import torch
import torch.nn as nn
import sys

from .prompts.imagenet_template import *
from .simfeatup_dev.upsamplers import get_upsampler
from .open_clip import tokenizer, create_model
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F


# from BLIP.models.blip_retrieval import blip_retrieval
# import gem



@MODELS.register_module()
class SegEarthSegmentation(BaseSegmentor):
    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=0.0,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 bg_idx=0,
                 feature_up=True,
                 feature_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='your/model/path')):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model("checkpoints/ViT-B-16.pt", pretrained='openai', precision='fp16')

        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.feature_up = feature_up
        self.cls_token_lambda = cls_token_lambda
        # self.output_cls_token = cls_token_lambda != 0
        self.output_cls_token = True
        self.bg_idx = bg_idx

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        else:
            self.patch_size = self.net.visual.patch_size

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad(): # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query =self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                           truncation=True, max_length=35,
                                           return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if feature_up:
            self.feat_dim = self.query_features.shape[-1]
            self.upsampler = get_upsampler(feature_up_cfg['model_name'], self.feat_dim).cuda().half()
            ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            self.upsampler.load_state_dict(weights_dict, strict=True)

        self.image_features = None

    def get_feature(self, img):
        if type(img) == list:
            img = img[0]
        H, W = img.shape[2:]
        self.img_shape = (H, W)
        pad = self.compute_padsize(H, W, self.patch_size[0])
        if any(pad):
            img = nn.functional.interpolate(img, (H+pad[2]+pad[3], W+pad[0]+pad[1]))

        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, self.output_cls_token)
        _, image_features = image_features
        feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
        return image_features

    def forward_feature(self, img, logit_size=None, **kwargs):
        if type(img) == list:
            img = img[0]
        H, W = img.shape[2:]
        self.img_shape = (H, W)
        pad = self.compute_padsize(H, W, self.patch_size[0])
        if any(pad):
            img = nn.functional.pad(img, pad)
            # img = nn.functional.interpolate(img, (H+pad[2]+pad[3], W+pad[0]+pad[1]))

        self.pad = pad
        self.img = img

        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, self.output_cls_token,
                                                   **kwargs)
            
        if self.output_cls_token:
            image_cls_token, image_features = image_features
            image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            cls_logits = image_cls_token @ self.query_features.T
            self.cls_logits = cls_logits


        # featup
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            self.image_features = image_features
            with torch.cuda.amp.autocast():
                image_features_up = self.upsampler(image_features, img).half()
            image_features_up_ = image_features_up.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)
        else:
            image_features_up_ = image_features
        image_features_up_norm = image_features_up_ / image_features_up_.norm(dim=-1, keepdim=True)
        logits = image_features_up_norm @ self.query_features.T

        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda
        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')


        if any(pad):
            l, t = pad[0], pad[2]
            logits = logits[:, :, t:t + H, l:l + W]
            # logits = nn.functional.interpolate(logits, (H, W))

        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224, **kwargs):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                # if 'hsi' in kwargs.keys():
                #     crop_img_hsi = kwargs['hsi'][y1:y2, x1:x2]
                #     crop_img_hsi = torch.from_numpy(crop_img_hsi).permute(2, 0, 1).unsqueeze(0).to(crop_img.device)
                # else:
                #     crop_img_hsi = None
                crop_img_hsi = None

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                # if any(pad):
                #     crop_img = nn.functional.pad(crop_img, pad)
                #     if crop_img_hsi is not None:
                #         crop_img_hsi = nn.functional.pad(crop_img_hsi, pad)

                # if crop_img_hsi is not None:
                #     crop_img = nn.functional.interpolate(crop_img, scale_factor=2)
                #     crop_img_hsi = nn.functional.interpolate(crop_img_hsi, scale_factor=2)
                crop_seg_logit = self.forward_feature(crop_img, hsi=crop_img_hsi)
                # if crop_img_hsi is not None:
                #     crop_seg_logit = nn.functional.interpolate(crop_seg_logit, scale_factor=0.5)
                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    @torch.no_grad()
    def predict(self, inputs, data_samples, slide=True, **kwargs):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        inputs = inputs.half()
        if self.slide_crop > 0 and slide:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop, **kwargs)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], **kwargs)
        seg_pred, seg_logits = self.postprocess_result(seg_logits, data_samples)
        seg_pred, seg_logits = seg_pred.data.cpu().numpy().squeeze(0).squeeze(0), \
                               seg_logits.data.cpu().numpy().squeeze(0)
        return seg_pred, seg_logits

    def postprocess_result(self, seg_logits_batch, data_samples):
        for i in range(len(seg_logits_batch)):
            seg_logits = seg_logits_batch[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
            seg_logits_batch[i] = seg_logits
        seg_pred_batch = seg_logits_batch.argmax(1, keepdim=True)
        return seg_pred_batch, seg_logits_batch

    def predict_plp(self, upsampler):
        image_features = self.image_features.detach().float()
        query_features = self.query_features.detach().float()
        img = self.img.detach().float()
        pad = self.pad
        H, W = self.img_shape

        image_w, image_h = img[0].shape[-2], img[0].shape[-1]
        image_features_up = upsampler(image_features, img)
        image_features_up_ = image_features_up.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)
        image_features_up_norm = image_features_up_ / image_features_up_.norm(dim=-1, keepdim=True)
        logits = image_features_up_norm @ query_features.T

        if self.output_cls_token:
            logits = logits + self.cls_logits * self.cls_token_lambda
        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if any(pad):
            l, t = pad[0], pad[2]
            logits = logits[:, :, t:t + H, l:l + W]
        return logits

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """



def get_cls_idx(name_sets):
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices