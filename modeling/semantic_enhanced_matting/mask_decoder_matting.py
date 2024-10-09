import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
import cv2
from detectron2.layers.batch_norm import NaiveSyncBatchNorm

from modeling.semantic_enhanced_matting.modeling import TwoWayTransformer, MaskDecoder
from modeling.decoder.detail_capture import Detail_Capture
from modeling.decoder.unet_detail_capture import DetailUNet
# from nnMorpho.binary_operators import erosion


# class GenTrimapTorch(object):
#     def __init__(self, max_kernal=200):
#         self.max_kernal = max_kernal
#         self.erosion_kernels = [None] + [torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))).float().cuda() for size in range(1, self.max_kernal)]

#     def __call__(self, mask, kernel_size):
        
#         fg_width = kernel_size
#         bg_width = kernel_size

#         fg_mask = mask
#         bg_mask = 1 - mask

#         fg_mask = erosion(fg_mask, self.erosion_kernels[fg_width], border='a')
#         bg_mask = erosion(bg_mask, self.erosion_kernels[bg_width], border='a')

#         trimap = torch.ones_like(mask) * 0.5
#         trimap[fg_mask == 1] = 1.0
#         trimap[bg_mask == 1] = 0.0

#         return trimap


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoderMatting(MaskDecoder):
    def __init__(
        self, 
        model_type, 
        checkpoint_path, 
        detail_capture, 
        mask_token_only, 
        norm_type = 'LN', 
        norm_mask_logits = False,
        with_trimap = False,
        min_kernel_size = 20,
        kernel_div = 10,
        concat_gen_trimap = False,
    ):
        super().__init__(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_multimask_outputs=3,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        assert norm_type in {'BN', 'LN', 'SyncBN'}
        if norm_type == 'BN':
            self.norm = torch.nn.BatchNorm2d
        elif norm_type == 'SyncBN':
            self.norm = NaiveSyncBatchNorm
        else:
            self.norm = LayerNorm2d

        # checkpoint_dict = {"vit_b":"pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
        #                    "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
        #                    'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        # checkpoint_path = checkpoint_dict[model_type]

        self.load_state_dict(torch.load(checkpoint_path))
        print("Matting Decoder init from SAM MaskDecoder")

        self.frozen_params_str = set()
        for n, p in self.named_parameters():
            p.requires_grad = False
            self.frozen_params_str.add(n)

        self.detail_capture = detail_capture
        self.mask_token_only = mask_token_only
        self.norm_mask_logits = norm_mask_logits

        transformer_dim = 256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1
        self.concat_gen_trimap = concat_gen_trimap

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            self.norm(transformer_dim),
            nn.GELU(), 
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )
        
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            self.norm(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
            self.norm(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1)
        )

        if isinstance(self.detail_capture, Detail_Capture):
            self.glue_layer_0 = nn.Conv2d(self.detail_capture.fus_channs[2], transformer_dim // 8, 3, 1, 1)
        else:
            assert isinstance(self.detail_capture, DetailUNet)

        self.trainable_params_str = set()
        for n, p in self.named_parameters():
            if p.requires_grad:
                self.trainable_params_str.add(n)

        self.with_trimap = with_trimap
        self.min_kernel_size = min_kernel_size
        self.kernel_div = kernel_div
        if self.with_trimap and not self.concat_gen_trimap:
            # self.gen_trimap = GenTrimapTorch()
            raise ValueError('Discard GenTrimapTorch')

        # self.trainable_params_str = {'detail_capture', 'hf_token', 'hf_mlp', 'compress_vit_feat', 'embedding_encoder', 'embedding_maskfeature', 'glue_layer_0'}
        # for n, p in self.named_parameters():
        #     if p.requires_grad:
        #         assert n.split('.')[0] in self.trainable_params_str

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        # hq_token_only: bool,
        interm_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
        images: torch.Tensor,
        hr_images_ori_h_w = None,
        return_alpha_logits = False,
        pred_trimap=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT [B, 64, 64, 768]

        # upsample image_embeddings x4.0 with detail_capture & embedding_encoder & compress_vit_feat
            # regard hq_features as condition
        if isinstance(self.norm_mask_logits, float):
            norm_hq_features = hq_features / self.norm_mask_logits
        elif self.norm_mask_logits:
            norm_hq_features = hq_features / torch.std(hq_features, dim=(1, 2, 3), keepdim=True)
        else:
            norm_hq_features = hq_features

        if hr_images_ori_h_w is not None:
            assert not isinstance(self.detail_capture, Detail_Capture) and hq_features.shape[-2] == hq_features.shape[-1] == 1024
            lr_images_before_pad_h_w = (1024 / max(hr_images_ori_h_w) * hr_images_ori_h_w[0], 1024 / max(hr_images_ori_h_w) * hr_images_ori_h_w[1])
            lr_images_before_pad_h_w = (int(lr_images_before_pad_h_w[0] + 0.5), int(lr_images_before_pad_h_w[1] + 0.5))
            norm_hq_features = F.interpolate(
                norm_hq_features[:, :, :lr_images_before_pad_h_w[0], :lr_images_before_pad_h_w[1]], 
                size = (images.shape[-2], images.shape[-1]), 
                mode = 'bilinear', 
                align_corners = False
            )
        
        if self.concat_gen_trimap:
            pred_trimap = F.interpolate(pred_trimap, size=(images.shape[-2], images.shape[-1]), mode='bilinear', align_corners=False)
            pred_trimap = torch.argmax(pred_trimap, dim=1, keepdim=True).float() / 2.0
            norm_hq_features = torch.concat((norm_hq_features, pred_trimap), dim=1)
        elif self.with_trimap:
            mask = (norm_hq_features > 0).float()
            for i_batch in range(image_embeddings.shape[0]):
                mask_area = torch.sum(mask[i_batch])
                kernel_size = max(self.min_kernel_size, int((mask_area ** 0.5) / self.kernel_div))
                kernel_size = min(kernel_size, self.gen_trimap.max_kernal - 1)
                mask[i_batch, 0] = self.gen_trimap(mask[i_batch, 0], kernel_size=kernel_size)
            trimaps = mask
            norm_hq_features = torch.concat((norm_hq_features, trimaps), dim=1)

        conditional_images = torch.concatenate((images, norm_hq_features), dim=1)  # [B, 4, 1024, 1024]

        if isinstance(self.detail_capture, Detail_Capture):
            detail_features = self.detail_capture.convstream(conditional_images)  # [B, 4, 1024, 1024] --> D0: [B, 4, 1024, 1024], D1: [B, 48, 512, 512], D2: [B, 96, 256, 256], D3: [B, 192, 128, 128]
            matting_features = self.detail_capture.fusion_blks[0](image_embeddings, detail_features['D3'])  # [B, 256, 64, 64] & [B, 192, 128, 128] --> [B, 256, 128, 128]
            matting_features = self.detail_capture.fusion_blks[1](matting_features, detail_features['D2'])  # [B, 256, 128, 128] & [B, 96, 256, 256] --> [B, 128, 256, 256]
            matting_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features) + self.glue_layer_0(matting_features)  # [B, 32, 256, 256]
        else:
            matting_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                matting_feature = matting_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks, 0)  # [B, 5, 256, 256]
        iou_preds = torch.cat(iou_preds, 0)  # [4, 4]

        if self.mask_token_only:
            masks_matting = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]  # [B, 1, 256, 256]
        else:
            masks_matting = masks  # [B, 5, 256, 256]

        if hr_images_ori_h_w is not None:
            vit_features = F.interpolate(
                vit_features[:, :, :math.ceil(lr_images_before_pad_h_w[0] / 16), :math.ceil(lr_images_before_pad_h_w[1] / 16)], 
                size = (images.shape[-2] // 16, images.shape[-1] // 16), 
                mode = 'bilinear', 
                align_corners = False
            )
            masks_matting = F.interpolate(
                masks_matting[:, :, :math.ceil(lr_images_before_pad_h_w[0] / 4), :math.ceil(lr_images_before_pad_h_w[1] / 4)], 
                size = (images.shape[-2] // 4, images.shape[-1] // 4), 
                mode = 'bilinear', 
                align_corners = False
            )

        if isinstance(self.detail_capture, Detail_Capture):
            matting_features = self.detail_capture.fusion_blks[2](masks_matting, detail_features['D1'])
            matting_features = self.detail_capture.fusion_blks[3](matting_features, detail_features['D0'])
            alpha = torch.sigmoid(self.detail_capture.matting_head(matting_features))
        else:
            if return_alpha_logits:
                output = self.detail_capture(conditional_images, vit_features, masks_matting, return_alpha_logits = True)
                alpha = torch.sigmoid(output[0]), output[1]
            else:
                alpha = torch.sigmoid(self.detail_capture(conditional_images, vit_features, masks_matting, return_alpha_logits = False))

        if hr_images_ori_h_w is not None:
            alpha = alpha[:, :, :hr_images_ori_h_w[0], :hr_images_ori_h_w[1]]
        
        return alpha
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        matting_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)  # [6, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)  # [1, 6, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # [1, 8, 256]

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)   # [1, 256, 64, 64]
        src = src + dense_prompt_embeddings  # [1, 256, 64, 64] + [1, 256, 64, 64]
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # [1, 256, 64, 64]
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  # [1, 8, 256], [1, 4096, 256]
        iou_token_out = hs[:, 0, :]  # [1, 256]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]  # [1, 5, 256]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # [1, 256, 64, 64]

        upscaled_embedding_sam = self.output_upscaling(src)  # [1, 32, 256, 256]
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + matting_feature  # [1, 32, 256, 256]
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)  # 5 * [1, 32] --> [1, 5, 32]
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 32] @ [1, 32, 65536] --> [1, 4, 256, 256]
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)  # [1, 1, 32] @ [1, 32, 65536] --> [1, 1, 256, 256]
        masks = torch.cat([masks_sam,masks_ours], dim=1)  # [1, 5, 256, 256]
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred