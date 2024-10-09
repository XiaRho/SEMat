import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from PIL import Image
from copy import deepcopy
from collections import defaultdict

from detectron2.structures import ImageList
from detectron2.utils.comm import get_local_rank
from modeling.semantic_enhanced_matting.predictor import SamPredictor
from modeling.semantic_enhanced_matting.condition_conv import ConditionConv, ConditionEmbedding, ConditionAdd, BBoxEmbedInteract, BBoxInteract, BBoxInteractInOut
from modeling.semantic_enhanced_matting.modeling.image_encoder import PatchEmbed
from modeling.semantic_enhanced_matting.modeling.common import LayerNorm2d
from modeling.decoder.unet_detail_capture import MattingDetailDecoder
from modeling.semantic_enhanced_matting.feature_fusion import FeatureFusion
from sam2.sam2_image_predictor import SAM2ImagePredictor

from modeling.semantic_enhanced_matting.modeling.mask_decoder_hq_matting import MaskDecoderHQMatting
from modeling.semantic_enhanced_matting.modeling import TwoWayTransformer

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer

from data.rand_augment import RandAugment
import random
import kornia.filters as kf


class SamHqMatte(nn.Module):

    target_length = 1024

    def __init__(
        self,
        *,
        sam_model,
        hq_token_only,
        hq_features_type,
        matting_decoder,
        criterion,
        pixel_mean,
        pixel_std,
        multimask_output=False,
        vis_period=None,
        output_dir=None,
        lora_rank = None,
        lora_alpha = None,
        lora_target_modules = ["qkv", "proj"],
        lora_dropout = 0.1,
        w_dora = False,
        w_rslora = False,
        lora_on_mask_decoder = False,
        frozen_sam_hq_reg = None,
        reg_margin = 0.85,
        w_attention_mask = False,
        alpha_reg_range = None,
        alpha_reg_weight = 1.0,
        coconut_pl = False,
        coconut_pl_alpha = 1.0,
        coconut_self_training = False,
        eval_w_sam_hq_mask = False,
        backbone_condition = False,
        condition_wo_conv = False,
        w_only_bbox_cond = False,
        coconut_only_known_l1 = False,
        backbone_bbox_prompt = None,
        backbone_bbox_prompt_loc = [2, 3], 
        backbone_bbox_prompt_loss_weight = 1.0,
        concat_gen_trimap = False,
        multi_matting_decoder = None,
        w_all_logits = False,
        bbox_prompt_all_block = None,
        matting_token = False,
        test_w_hq_token = False,
        sam_hq_token_reg = None,
        feat_cross_attn_fusion = False,
        trimap_loss_type = None,
        reg_on_sam_logits = False,
        reg_w_bce_loss = False,
        complex_trimap_pred_layer = False,
        matting_token_sup = None,
        matting_token_sup_loss_weight = None,
        sam2 = False,
    ):
        super(SamHqMatte, self).__init__()

        self.sam_model = sam_model
        self.sam_predictor = SamPredictor(self.sam_model) if not sam2 else SAM2ImagePredictor(self.sam_model)  # already in eval mode and no_grad
        self.hq_token_only = hq_token_only
        self.multimask_output = multimask_output
        self.hq_features_type = hq_features_type

        self.matting_decoder = matting_decoder

        self.criterion = criterion

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.vis_period = vis_period
        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'vis_results')
            os.makedirs(self.output_dir, exist_ok=True)
        self.train_iter_index = 0

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.w_dora = w_dora
        self.w_rslora = w_rslora
        self.lora_on_mask_decoder = lora_on_mask_decoder
        self.frozen_sam_hq_reg = frozen_sam_hq_reg
        self.reg_margin = reg_margin
        self.w_attention_mask = w_attention_mask
        self.alpha_reg_range = alpha_reg_range
        self.alpha_reg_weight = alpha_reg_weight
        self.coconut_pl = coconut_pl
        self.coconut_pl_alpha = coconut_pl_alpha
        self.coconut_self_training = coconut_self_training
        self.eval_w_sam_hq_mask = eval_w_sam_hq_mask
        self.backbone_condition = backbone_condition
        self.condition_wo_conv = condition_wo_conv
        self.w_only_bbox_cond = w_only_bbox_cond
        self.coconut_only_known_l1 = coconut_only_known_l1
        self.backbone_bbox_prompt = backbone_bbox_prompt
        self.backbone_bbox_prompt_loc = backbone_bbox_prompt_loc
        self.backbone_bbox_prompt_loss_weight = backbone_bbox_prompt_loss_weight
        self.concat_gen_trimap = concat_gen_trimap
        self.multi_matting_decoder = multi_matting_decoder
        self.w_all_logits = w_all_logits
        self.bbox_prompt_all_block = bbox_prompt_all_block
        self.matting_token = matting_token
        self.test_w_hq_token = test_w_hq_token
        self.sam_hq_token_reg = sam_hq_token_reg
        self.feat_cross_attn_fusion = feat_cross_attn_fusion
        self.trimap_loss_type = trimap_loss_type
        self.reg_on_sam_logits = reg_on_sam_logits
        self.reg_w_bce_loss = reg_w_bce_loss
        self.complex_trimap_pred_layer = complex_trimap_pred_layer
        self.matting_token_sup = matting_token_sup
        self.sam2 = sam2
        assert self.matting_token_sup in {'alpha', 'trimap', None}
        self.matting_token_sup_loss_weight = matting_token_sup_loss_weight
        if self.matting_token_sup is not None:
            assert self.backbone_bbox_prompt in {'bbox', None}
        if self.frozen_sam_hq_reg is not None:
            assert self.lora_rank is not None
        if self.w_attention_mask:
            self.attention_head = deepcopy(self.matting_decoder)
        if self.coconut_self_training:
            self.rand_aug = RandAugment(3,6)
            self.warm_iter_coconut_self_training = 5000
        if self.backbone_condition:
            assert self.lora_rank is not None
        if self.backbone_bbox_prompt is not None:
            assert self.lora_rank is not None
        if self.w_all_logits:
            self.sam_predictor.model.mask_decoder.w_all_logits = True
        if self.bbox_prompt_all_block:
            assert self.lora_rank is not None
        if self.matting_token and not self.sam2:
            self.sam_predictor.model.mask_decoder.hq_token_only = self.hq_token_only

    @property
    def device(self):
        return self.pixel_mean.device

    def init_lora(self, model=None):
        if model is not None and self.lora_rank >= 1:
            if self.lora_on_mask_decoder:
                self.lora_target_modules += ["q_proj", "k_proj", "v_proj", "out_proj"]
                modules_to_save = None
            else:
                modules_to_save = ['matting_decoder']

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                use_rslora=self.w_rslora,
                use_dora=self.w_dora,
                init_lora_weights="gaussian",
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, lora_config)
            if self.lora_on_mask_decoder:
                for n, p in model.matting_decoder.named_parameters():
                    if n.split('modules_to_save.default.')[-1] in model.matting_decoder.trainable_params_str:
                        p.requires_grad = True
            else:
                for n, p in model.matting_decoder.named_parameters():
                    if n.split('modules_to_save.default.')[-1] in model.matting_decoder.frozen_params_str:
                        p.requires_grad = False
            return model
        elif self.lora_rank >= 1:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                use_rslora=self.w_rslora,
                use_dora=self.w_dora,
                init_lora_weights="gaussian",
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
            )
            self.sam_predictor.model.image_encoder = get_peft_model(self.sam_predictor.model.image_encoder, lora_config)

            if self.sam2:
                for n, p in self.sam_predictor.model.image_encoder.named_parameters():
                    if 'bbox_mask' in n:
                        p.requires_grad = True

        if self.backbone_condition:
            if self.w_only_bbox_cond:
                self.condition_embedding = ConditionEmbedding(condition_num = 4, pos_embedding_dim = 160)
            else:
                self.condition_embedding = ConditionEmbedding(condition_num = 5, pos_embedding_dim = 128)

            if self.condition_wo_conv:
                self.condition_conv = nn.ModuleList([ConditionAdd() for _ in range(4)])
            else:
                self.condition_conv = nn.ModuleList([ConditionConv(
                    in_channels = self.sam_predictor.model.image_encoder.embed_dim, 
                    out_channels = self.sam_predictor.model.image_encoder.embed_dim,
                    bottleneck_channels = 512
                ) for _ in range(4)])
        
        if self.backbone_bbox_prompt is not None and not self.sam2:
            self.condition_layer = nn.ModuleDict()
            self.condition_layer['patch_embed'] =  PatchEmbed(
                kernel_size=(self.sam_predictor.model.image_encoder.patch_size, self.sam_predictor.model.image_encoder.patch_size),
                stride=(self.sam_predictor.model.image_encoder.patch_size, self.sam_predictor.model.image_encoder.patch_size),
                in_chans=4,
                embed_dim=self.sam_predictor.model.image_encoder.embed_dim,
            )
            if self.multi_matting_decoder is None:
                if self.backbone_bbox_prompt in {'trimap', 'alpha_trimap'}:
                    transformer_dim = self.sam_predictor.model.image_encoder.embed_dim
                    for i in self.backbone_bbox_prompt_loc:
                        if self.complex_trimap_pred_layer:
                            self.condition_layer['{}_pred_layer'.format(i)] = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 2),  # 512
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 2, transformer_dim // 4, kernel_size=3, stride=1, padding=1),
                                LayerNorm2d(transformer_dim // 4),  # 256
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 8),  # 128
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 8, transformer_dim // 16, kernel_size=3, stride=1, padding=1),
                                LayerNorm2d(transformer_dim // 16),  # 64
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 16, 3, kernel_size=3, stride=1, padding=1),
                            )
                        else:
                            self.condition_layer['{}_pred_layer'.format(i)] = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 4),
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 8, 3, kernel_size=1, stride=1),
                            )
                elif self.backbone_bbox_prompt == 'alpha':
                    transformer_dim = self.sam_predictor.model.image_encoder.embed_dim
                    for i in self.backbone_bbox_prompt_loc:
                        if self.complex_trimap_pred_layer:
                            self.condition_layer['{}_pred_layer'.format(i)] = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 2),  # 512
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 2, transformer_dim // 4, kernel_size=3, stride=1, padding=1),
                                LayerNorm2d(transformer_dim // 4),  # 256
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 8),  # 128
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 8, transformer_dim // 16, kernel_size=3, stride=1, padding=1),
                                LayerNorm2d(transformer_dim // 16),  # 64
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 16, 1, kernel_size=3, stride=1, padding=1),
                                nn.Sigmoid()
                            )
                        else:
                            self.condition_layer['{}_pred_layer'.format(i)] = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 4),
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                nn.GELU(),
                                nn.Conv2d(transformer_dim // 8, 1, kernel_size=1, stride=1),
                                nn.Sigmoid()
                            )
            if self.bbox_prompt_all_block is not None:
                if self.bbox_prompt_all_block == 'reuse_cross-self-attn':
                    self.condition_layer['prompt_layer'] = BBoxInteract(
                        position_point_embedding = deepcopy(self.sam_predictor.model.prompt_encoder.pe_layer), 
                        point_weight = deepcopy(self.sam_predictor.model.prompt_encoder.point_embeddings)
                    )
                elif self.bbox_prompt_all_block == 'in-out-bbox_cross-self-attn':
                    self.condition_layer['prompt_layer'] = BBoxInteractInOut(downsample_rate = 2)
                else:
                    embed_type, interact_type = self.bbox_prompt_all_block.split('_')
                    self.condition_layer['prompt_layer'] = BBoxEmbedInteract(embed_type, interact_type)

        if self.feat_cross_attn_fusion:
            self.condition_layer['feature_fusion'] = FeatureFusion(in_channels=self.sam_predictor.model.image_encoder.embed_dim, attn_compression_ratio=8)

    def condition_bbox_and_instance_num(self):
        self.sam_predictor.model.image_encoder.conv_necks = None

    def forward_samhq_and_matting_decoder(self, images, bbox, condition_proj=None, return_hq_token=False):
        # get features from SAM image encoder
        if self.sam2:
            interm_features, sam2_logits, matting_logits, pred_trimap = self.forward_samhq(images, bbox, condition_proj)
            sam2_logits = F.interpolate(sam2_logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
            matting_logits = F.interpolate(matting_logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
            sam_hq_matting_token = {
                'masks_hq': sam2_logits,
                'masks_matting': matting_logits
            }
            hq_features = matting_logits
            low_res_masks = matting_logits
        else:
            if self.matting_token:
                features, image_pe, sparse_embeddings, dense_embeddings, interm_features, sam_hq_matting_token, pred_trimap = self.forward_samhq(images, bbox, condition_proj)
                if return_hq_token:
                    return sam_hq_matting_token['masks_hq']
                else:
                    if not self.training and self.test_w_hq_token:
                        low_res_masks, hq_features = sam_hq_matting_token['masks_hq'], sam_hq_matting_token['masks_hq']
                    else:
                        low_res_masks, hq_features = sam_hq_matting_token['masks_matting'], sam_hq_matting_token['masks_matting']
            else:
                features, image_pe, sparse_embeddings, dense_embeddings, interm_features, hq_features, sam_logits, low_res_masks, pred_trimap = self.forward_samhq(images, bbox, condition_proj)
                if return_hq_token:
                    return hq_features
                sam_hq_matting_token = {'masks_hq': hq_features, 'masks_sam': sam_logits}

        # get alpha from our proposed matting_decoder
        if isinstance(self.matting_decoder, MattingDetailDecoder):
            pred_alpha = self.matting_decoder(
                images = images,
                hq_features = hq_features,
                vit_intern_feat = interm_features,
                return_alpha_logits = (self.alpha_reg_range is not None),
                pred_trimap = pred_trimap
            )
        else:
            pred_alpha = self.matting_decoder(
                image_embeddings = features,  # [B, 256, 64, 64]
                image_pe = image_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
                multimask_output = False,
                interm_embeddings = interm_features,  # [B, 256, 64, 64]
                hq_features = hq_features,
                images = images,
                return_alpha_logits = (self.alpha_reg_range is not None),
                pred_trimap = pred_trimap
            )
        return low_res_masks, pred_alpha, pred_trimap, sam_hq_matting_token

    def forward(self, batched_inputs):  # image: [1, 3, 643, 960]: 0.0~1.0, trimap: [1, 1, 643, 960]: 0.0~1.0

        inputs = self.preprocess_inputs(batched_inputs) 
        images, bbox, gt_alpha, trimap, condition = inputs['images'], inputs['bbox'], inputs['alpha'], inputs['trimap'], inputs['condition']

        if self.backbone_condition:
            condition_proj = self.condition_embedding(condition) 
        elif self.backbone_bbox_prompt is not None or self.bbox_prompt_all_block is not None:
            condition_proj = bbox
        else:
            condition_proj = None

        low_res_masks, pred_alpha, pred_trimap, sam_hq_matting_token = self.forward_samhq_and_matting_decoder(images, bbox, condition_proj)
        
        assert not self.training
        if self.eval_w_sam_hq_mask:
            self.sam_predictor.model.image_encoder.disable_adapter_layers()
            with torch.no_grad():
                ori_features, ori_interm_features = self.sam_predictor.model.image_encoder(images)
                samhq_low_res_masks = self.forward_samhq_others(images, bbox, ori_features, ori_interm_features)[-1]
                samhq_low_res_masks = F.interpolate(samhq_low_res_masks, size=(images.shape[-2], images.shape[-1]), mode='bilinear', align_corners=False)
            self.sam_predictor.model.image_encoder.enable_adapter_layers()

            return pred_alpha, samhq_low_res_masks
        else:
            return pred_alpha
        
    def forward_samhq_image_encoder(self, images, condition_proj=None):
        if self.sam2:
            backbone_out = self.sam_predictor.model.forward_image([images, condition_proj])
            _, vision_feats, _, _ = self.sam_predictor.model._prepare_backbone_features(backbone_out)
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.sam_predictor.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam_predictor.model.no_mem_embed
            feats = [
                feat.permute(1, 2, 0).view(feat.shape[1], -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.sam_predictor._bb_feat_sizes[::-1])
            ][::-1]
            return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}, None, None
        else:
            if self.backbone_condition:
                condition_layer = self.condition_conv 
            elif self.backbone_bbox_prompt:
                condition_layer = self.condition_layer
            else:
                condition_layer = None
            # [B, 3, 1024, 1024]: -2. ~ 2. --> [B, 256, 64, 64], 4 x [B, 64, 64, 768]
            features, interm_features, pred_trimap = self.sam_predictor.model.image_encoder(images, condition_proj, condition_layer)
            return features, interm_features, pred_trimap
    
    # @torch.no_grad()
    def forward_samhq_others(self, images, bbox, features, interm_features):
        if self.sam2:
            sam2_logits, matting_logits = self.sam_predictor.predict_batch_boxes_and_features(bbox, features)
            return features, sam2_logits, matting_logits
        
        image_pe = self.sam_predictor.model.prompt_encoder.get_dense_pe()

        cat_sparse_embeddings = []
        cat_dense_prompt_embeddings = []
        cat_hq_features = []
        cat_sam_logits = []
        cat_low_res_masks = []
        cat_sam_hq_matting_token = defaultdict(list)

        for idx in range(images.shape[0]):
            # get hq_features from SAM_HQ mask decoder

                # Embed prompts
            sparse_embeddings, dense_embeddings = self.sam_predictor.model.prompt_encoder(
                points=None,
                # boxes=bbox[idx: idx + 1],
                boxes=bbox[idx],  # [N, 4]
                masks=None,
            )  # [B, 2, 256], [B, 256, 64, 64]

                # Predict masks
            if isinstance(self.sam_predictor.model.mask_decoder, MaskDecoderHQMatting):
                sam_hq_matting_token = self.sam_predictor.model.mask_decoder(
                    image_embeddings = features[idx: idx + 1],
                    image_pe = image_pe,
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = self.multimask_output,
                    interm_embeddings = [interm_feature[idx: idx + 1] for interm_feature in interm_features],
                )
                for key in sam_hq_matting_token.keys():
                    cat_sam_hq_matting_token[key].append(sam_hq_matting_token[key])
            else:
                low_res_masks, masks_sam, hq_features = self.sam_predictor.model.mask_decoder(
                    image_embeddings = features[idx: idx + 1],
                    image_pe = image_pe,
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings = dense_embeddings,
                    multimask_output = self.multimask_output,
                    hq_token_only = self.hq_token_only,
                    interm_embeddings = [interm_feature[idx: idx + 1] for interm_feature in interm_features],
                    return_hq_features_type = self.hq_features_type
                )
                cat_hq_features.append(hq_features)
                cat_sam_logits.append(masks_sam)
                cat_low_res_masks.append(low_res_masks)

            cat_sparse_embeddings.append(sparse_embeddings)
            cat_dense_prompt_embeddings.append(dense_embeddings)
            
        sparse_embeddings = torch.stack(cat_sparse_embeddings, dim=0)  # [B, 1, 2, 256]
        dense_embeddings = torch.stack(cat_dense_prompt_embeddings, dim=0)  # [B, 1, 256, 64, 64]
        
        if self.matting_token:
            for key in cat_sam_hq_matting_token.keys():
                cat_sam_hq_matting_token[key] = torch.cat(cat_sam_hq_matting_token[key], dim=0)
                cat_sam_hq_matting_token[key] = F.interpolate(cat_sam_hq_matting_token[key], size=images.shape[-2:], mode='bilinear', align_corners=False)
            sam_hq_matting_token = cat_sam_hq_matting_token
            return features, image_pe, sparse_embeddings, dense_embeddings, interm_features, sam_hq_matting_token
        else:
            hq_features = torch.cat(cat_hq_features, dim=0)  # [B, 1, 256, 256]
            low_res_masks = torch.cat(cat_low_res_masks, dim=0)  # [B, 1, 256, 256]
            hq_features = F.interpolate(hq_features, size=images.shape[-2:], mode='bilinear', align_corners=False)  # [B, 1, 256, 256] --> [B, 1, 1024, 1024]
            sam_logits = torch.cat(cat_sam_logits, dim=0)
            sam_logits = F.interpolate(sam_logits, size=images.shape[-2:], mode='bilinear', align_corners=False)  # [B, 1, 256, 256] --> [B, 1, 1024, 1024]
            return features, image_pe, sparse_embeddings, dense_embeddings, interm_features, hq_features, sam_logits, low_res_masks

    def forward_samhq(self, images, bbox, condition_proj=None):
        if self.lora_rank is None:
            with torch.no_grad():
                features, interm_features, pred_trimap = self.forward_samhq_image_encoder(images, condition_proj)
        else:
            features, interm_features, pred_trimap = self.forward_samhq_image_encoder(images, condition_proj)

        return self.forward_samhq_others(images, bbox, features, interm_features) + (pred_trimap, )

    def get_frozen_sam_logits(self, images, bbox, mask_type='hq'):
        
        if self.sam2:
            features, _, _ = self.forward_samhq_image_encoder(images)
            sam2_logits = self.sam_predictor.predict_batch_boxes_and_features(bbox, features, wo_matting_token=True)
            sam2_logits = F.interpolate(sam2_logits, size=images.shape[-2:], mode='bilinear', align_corners=False)
            return sam2_logits

        assert mask_type in {'hq', 'sam'} 
        features, interm_features, _ = self.forward_samhq_image_encoder(images)
        image_pe = self.sam_predictor.model.prompt_encoder.get_dense_pe()

        cat_logits = []
        for idx in range(images.shape[0]):
            sparse_embeddings, dense_embeddings = self.sam_predictor.model.prompt_encoder(points=None, boxes=bbox[idx], masks=None)

            low_res_masks, masks_sam, hq_features = self.sam_predictor.model.frozen_mask_decoder(
                image_embeddings = features[idx: idx + 1],
                image_pe = image_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
                multimask_output = self.multimask_output,
                hq_token_only = self.hq_token_only,
                interm_embeddings = [interm_feature[idx: idx + 1] for interm_feature in interm_features],
                return_hq_features_type = self.hq_features_type
            )
            if mask_type == 'hq':
                cat_logits.append(hq_features)  
            else:
                cat_logits.append(masks_sam)  
        
        logits = torch.cat(cat_logits, dim=0)  # [B, 1, 256, 256]
        logits = F.interpolate(logits, size=images.shape[-2:], mode='bilinear', align_corners=False)  # [B, 1, 256, 256] --> [B, 1, 1024, 1024]
        return logits

    def vis_training_results(self, **kwargs):
        # images, bbox, trimap, low_res_masks, pred_alpha, alpha
        self.train_iter_index += 1
        if self.train_iter_index % self.vis_period == 0:
            batch_save_results = []
            save_path = os.path.join(self.output_dir, '{:06d}_rank{}.jpg'.format(self.train_iter_index, get_local_rank()))
            
            # [('images', (4, 3, 1024, 1024), -2.117904, 2.64), ('bbox', (4, 1, 4), 0.0, 1023.0), ('trimap', (4, 1, 1024, 1024), 0.0, 1.0), ('low_res_masks', (4, 1, 256, 256), -20.38, 10.15), ('pred_alpha', (4, 1, 1024, 1024), 0.1547, 0.791), ('alpha', (4, 1, 1024, 1024), 0.0, 1.0)]
            for key in kwargs.keys():
                if key == 'bbox':
                    continue
                # turn all tensor to [B, H, W, 3]: 0~255 np.int8
                if key == 'images':
                    kwargs[key] = kwargs[key] * self.pixel_std + self.pixel_mean
                    kwargs[key] = kwargs[key].permute(0, 2, 3, 1) * 255.0
                    for i in range(kwargs['images'].shape[0]):
                        l, u, r, d = int(kwargs['bbox'][i, 0, 0].item()), int(kwargs['bbox'][i, 0, 1].item()), int(kwargs['bbox'][i, 0, 2].item()), int(kwargs['bbox'][i, 0, 3].item())
                        red_line = torch.tensor([[255., 0., 0.]], device=kwargs[key].device, dtype=kwargs[key].dtype)
                        kwargs[key][i, u: d, l, :] = red_line
                        kwargs[key][i, u: d, r, :] = red_line
                        kwargs[key][i, u, l: r, :] = red_line
                        kwargs[key][i, d, l: r, :] = red_line
                elif key in {'low_res_masks', 'frozen_hq_token'}:
                    if torch.max(kwargs[key]) <= 1:  # coconut ori alpha
                        kwargs[key] = kwargs[key].permute(0, 2, 3, 1).repeat(1, 1, 1, 3) * 255.0
                    else:
                        kwargs[key] = F.interpolate(kwargs[key], size=(kwargs['images'].shape[-3], kwargs['images'].shape[-2]), mode='bilinear', align_corners=False)
                        kwargs[key] = (kwargs[key] > self.sam_predictor.model.mask_threshold).float().permute(0, 2, 3, 1).repeat(1, 1, 1, 3) * 255.0
                else:
                    kwargs[key] = kwargs[key].permute(0, 2, 3, 1).repeat(1, 1, 1, 3) * 255.0

                kwargs[key] = np.uint8(kwargs[key].detach().cpu().numpy())

            for i in range(kwargs['images'].shape[0]):
                save_results = []
                for key in kwargs.keys():
                    if key != 'bbox':
                        save_results.append(kwargs[key][i])
                batch_save_results.append(np.concatenate(save_results, axis=1))
            
            Image.fromarray(np.concatenate(batch_save_results, axis=0)).save(save_path)

    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        output = dict()

        if "alpha" in batched_inputs:
            alpha = batched_inputs["alpha"].to(self.device)
        else:
            alpha = None

        bbox = batched_inputs["bbox"].to(self.device)

        if self.training and self.coconut_self_training and sum([i == 'COCONut' for i in batched_inputs['dataset_name']]) >= 1:
            output['coconut_ori_img'] = []
            output['coconut_trimap'] = []
            output['coconut_bbox'] = []
            output['coconut_idx'] = []
            for i, dataset_name in enumerate(batched_inputs['dataset_name']):
                if dataset_name == 'COCONut':
                    # generate coconut_aug_img
                    img_np = np.uint8(batched_inputs["image"][i].permute(1, 2, 0).cpu().numpy() * 255.)
                    strong_aug_img = self.rand_aug(Image.fromarray(img_np), cutout = False)
                    strong_aug_img_tensor = torch.from_numpy(np.array(strong_aug_img)).to(self.device).permute(2, 0, 1)[None] / 255.
                    blur_kernel_sigma = 1.0 + random.random()  # random from 1.0 ~ 2.0
                    blur_filter = kf.GaussianBlur2d((101, 101), (blur_kernel_sigma, blur_kernel_sigma))
                    blur_strong_aug_img_tensor = blur_filter(strong_aug_img_tensor)[0]

                    output['coconut_ori_img'].append(batched_inputs["image"][i])
                    batched_inputs["image"][i] = blur_strong_aug_img_tensor

                    # generate coconut_trimap
                    coconut_mask = (alpha[i] != 0).float()
                    mask_area = torch.sum(coconut_mask)
                    kernel_size = max(self.matting_decoder.min_kernel_size, int((mask_area ** 0.5) / 7))  # self.matting_decoder.kernel_div
                    kernel_size = min(kernel_size, self.matting_decoder.gen_trimap.max_kernal - 1)
                    output['coconut_trimap'].append(self.matting_decoder.gen_trimap(coconut_mask[0], kernel_size=kernel_size)[None])

                    output['coconut_bbox'].append(bbox[i])
                    output['coconut_idx'].append(i)

            output['coconut_ori_img'] = torch.stack(output['coconut_ori_img']).to(self.device)
            output['coconut_ori_img'] = (output['coconut_ori_img'] - self.pixel_mean) / self.pixel_std
            output['coconut_trimap'] = torch.stack(output['coconut_trimap']).to(self.device)
            output['coconut_bbox'] = torch.stack(output['coconut_bbox']).to(self.device)

        images = batched_inputs["image"].to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std
        assert images.shape[-2] == images.shape[-1] == 1024

        if 'trimap' in batched_inputs.keys():
            trimap = batched_inputs["trimap"].to(self.device)
            assert len(torch.unique(trimap)) <= 3
        else:
            trimap = None

        output['images'] = images
        output['bbox'] = bbox
        output['alpha'] = alpha
        output['trimap'] = trimap

        if 'hr_images' in batched_inputs.keys():
            hr_images = batched_inputs["hr_images"].to(self.device)
            hr_images = (hr_images - self.pixel_mean) / self.pixel_std
            _, _, H, W = hr_images.shape
            if hr_images.shape[-1] % 16 != 0 or hr_images.shape[-2] % 16 != 0:
                new_H = (16 - hr_images.shape[-2] % 16) + H if hr_images.shape[-2] % 16 != 0 else H
                new_W = (16 - hr_images.shape[-1] % 16) + W if hr_images.shape[-1] % 16 != 0 else W
                new_hr_images = torch.zeros((hr_images.shape[0], hr_images.shape[1], new_H, new_W)).to(self.device)
                new_hr_images[:,:,:H,:W] = hr_images[:,:,:,:]
                del hr_images
                hr_images = new_hr_images
            output['hr_images'] = hr_images
            output['hr_images_ori_h_w'] = (H, W)

        if 'dataset_name' in batched_inputs.keys():
            output['dataset_name'] = batched_inputs["dataset_name"]

        if self.backbone_condition:
            if self.w_only_bbox_cond:
                output['condition'] = output['bbox'][:, 0, :]
            else:
                multi_fg_float = batched_inputs["multi_fg"].to(bbox.device).float()[:, None] * 512
                output['condition'] = torch.concat((output['bbox'][:, 0, :], multi_fg_float), dim=-1)
        else:
            output['condition'] = None

        return output
