# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoderHQ, PromptEncoder, Sam, TwoWayTransformer, TinyViT
from .modeling.mask_decoder_hq_matting import MaskDecoderHQMatting


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, matting_token=0, wo_hq=False, frozen_decoder=False, mask_matting_res_add=True):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        matting_token=matting_token,
        wo_hq=wo_hq,
        frozen_decoder=frozen_decoder,
        mask_matting_res_add=mask_matting_res_add
    )


def build_sam_vit_b(checkpoint=None, matting_token=False, wo_hq=False, frozen_decoder=False):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        matting_token=matting_token,
        wo_hq=wo_hq,
        frozen_decoder=frozen_decoder
    )


def build_sam_vit_t(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
            image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            ),
            mask_decoder=MaskDecoderHQ(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                vit_dim=160,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(f, map_location=device)
        info = mobile_sam.load_state_dict(state_dict, strict=False)
        print(info)
    for n, p in mobile_sam.named_parameters():
        if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
            p.requires_grad = False
    return mobile_sam

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_tiny": build_sam_vit_t
}

def sam_model_registry_def(model_type, checkpoint, matting_token = 0, wo_hq = False, frozen_decoder = False, mask_matting_res_add=True):
    assert model_type in {"default", "vit_h", "vit_l", "vit_b", "vit_tiny"}
    return sam_model_registry[model_type](checkpoint=checkpoint, matting_token=matting_token, wo_hq=wo_hq, frozen_decoder=frozen_decoder, mask_matting_res_add=mask_matting_res_add)

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    matting_token=0,
    wo_hq=False,
    frozen_decoder=False,
    mask_matting_res_add=True
):
    # no_res_add only work when wo_hq and have mat ting token
    if not mask_matting_res_add:
        assert matting_token > 0

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    if matting_token > 0:
        mask_decoder = MaskDecoderHQMatting(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
            wo_hq=wo_hq,
            matting_token_num=matting_token,
            mask_matting_res_add=mask_matting_res_add
        )
    else:
        mask_decoder = MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
            wo_hq=wo_hq
        )
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(f, map_location=device)
        info = sam.load_state_dict(state_dict, strict=False)
        print(info)

    if frozen_decoder and checkpoint is not None:
        sam.frozen_mask_decoder = MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
            wo_hq=wo_hq
        )
        sam.frozen_mask_decoder.eval()
        info = sam.frozen_mask_decoder.load_state_dict({key.split('mask_decoder.')[1]: val for key, val in state_dict.items() if 'mask_decoder.' in key}, strict=False)
        print('load frozen_mask_decoder', info)
        # for n, p in sam.frozen_mask_decoder.named_parameters():
        #     p = state_dict['mask_decoder.' + n]

    for n, p in sam.named_parameters():
        # if 'hf_token' not in n and 'hf_mlp' not in n and 'compress_vit_feat' not in n and 'embedding_encoder' not in n and 'embedding_maskfeature' not in n:
        #     p.requires_grad = False
        if 'matting' not in n:
            p.requires_grad = False
        # p.requires_grad = False

    return sam
