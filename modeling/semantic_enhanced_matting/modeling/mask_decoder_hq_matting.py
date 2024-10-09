# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .mask_decoder_hq import MaskDecoderHQ, MLP


class MaskDecoderHQMatting(MaskDecoderHQ):
    def __init__(
        self,
        hq_token_only=False,
        matting_token_num=1,
        mask_matting_res_add=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.hq_token_only = hq_token_only
        self.matting_token_num = matting_token_num
        self.mask_matting_res_add = mask_matting_res_add
        if not self.mask_matting_res_add:
            assert self.wo_hq

        # Matting token parameters
        self.matting_hf_token = nn.Embedding(self.matting_token_num, self.transformer_dim) # Matting-Ouptput-Token
        self.matting_hf_mlp = MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3) # corresponding new MLP layer for Matting-Ouptput-Token
        self.num_mask_tokens = self.num_mask_tokens + self.matting_token_num
        
        # three conv fusion layers for obtaining Matting-Feature
        self.matting_compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(self.vit_dim, self.transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(self.transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 8, kernel_size=2, stride=2))
        
        self.matting_embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(self.transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2),
                                    )
        self.matting_embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(self.transformer_dim // 8, self.transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(self.transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(self.transformer_dim // 4, self.transformer_dim // 8, 3, 1, 1))


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
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
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        if not self.wo_hq:
            hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        else:
            hq_features = None
        matting_hq_features = self.matting_embedding_encoder(image_embeddings) + self.matting_compress_vit_feat(vit_features)

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
            matting_hq_features=matting_hq_features
        )

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            if not self.wo_hq:
                mask_slice = slice(1,self.num_mask_tokens - (self.matting_token_num + 1))  # matting_token_num + hq_token_num
            else:
                mask_slice = slice(1,self.num_mask_tokens - self.matting_token_num)  # matting_token_num
            iou_pred = iou_pred[:, mask_slice]
            iou_pred, max_iou_idx = torch.max(iou_pred,dim=1)
            iou_pred = iou_pred.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:,mask_slice]
            masks_sam = masks[:,mask_slice]

        if not self.wo_hq:
            masks_hq = masks[:,slice(self.num_mask_tokens - (self.matting_token_num + 1), self.num_mask_tokens - self.matting_token_num)]
        masks_matting = masks[:,slice(self.num_mask_tokens - self.matting_token_num, self.num_mask_tokens)]

        if not self.wo_hq:
            if self.hq_token_only:
                # masks_hq += masks_sam
                masks_matting += masks_hq
            else:
                masks_hq += masks_sam
                masks_matting += masks_hq 
        else:
            masks_hq = masks_sam
            if self.mask_matting_res_add:
                masks_matting = masks_sam + masks_matting
            else:
                masks_matting = masks_matting
        # Prepare output
        return {'masks_sam': masks_sam, 'masks_hq': masks_hq, 'masks_matting': masks_matting}

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
        matting_hq_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        if not self.wo_hq:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight, self.matting_hf_token.weight], dim=0)
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.matting_hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        if not self.wo_hq:
            upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b,1,1,1)
        upscaled_embedding_matting_hq = self.matting_embedding_maskfeature(upscaled_embedding_sam) + matting_hq_features.repeat(b,1,1,1)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - (self.matting_token_num + 1) or (self.wo_hq and i < self.num_mask_tokens - self.matting_token_num):
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            elif i == self.num_mask_tokens - (self.matting_token_num + 1):
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.matting_hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        if not self.wo_hq:
            masks_sam = (hyper_in[:,:self.num_mask_tokens - (self.matting_token_num + 1)] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
            masks_sam_hq = (hyper_in[:,self.num_mask_tokens - (self.matting_token_num + 1) : self.num_mask_tokens - self.matting_token_num] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        else:
            masks_sam = (hyper_in[:,:self.num_mask_tokens - self.matting_token_num] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_matting_hq = (hyper_in[:, self.num_mask_tokens - self.matting_token_num:] @ upscaled_embedding_matting_hq.view(b, c, h * w)).view(b, -1, h, w)

        if not self.wo_hq:
            masks = torch.cat([masks_sam, masks_sam_hq, masks_sam_matting_hq],dim=1)
        else:
            masks = torch.cat([masks_sam, masks_sam_matting_hq],dim=1)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
