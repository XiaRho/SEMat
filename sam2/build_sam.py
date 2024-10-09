# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    bbox_mask_matting_token = False,
    matting_logits_res_add = False,
    upscaled_embedding_res_add = True,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    
    if bbox_mask_matting_token:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.mask_decoder_matting_token=true",
            "++model.image_encoder.trunk._target_=sam2.modeling.backbones.hieradet.HieraBBoxMask",
            "++model.matting_logits_res_add=true" if matting_logits_res_add else "++model.matting_logits_res_add=false",
            "++model.upscaled_embedding_res_add=true" if upscaled_embedding_res_add else "++model.upscaled_embedding_res_add=false",
        ]

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, add_new_layer_weights=True)

    model = model.to(device)
    if mode == "eval":
        model.eval()
    
    if bbox_mask_matting_token:
        for n, p in model.named_parameters():
            if 'matting' in n or 'bbox_mask' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):

    from huggingface_hub import hf_hub_download

    model_id_to_filenames = {
        "facebook/sam2-hiera-tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
        "facebook/sam2-hiera-small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
        "facebook/sam2-hiera-base-plus": (
            "sam2_hiera_b+.yaml",
            "sam2_hiera_base_plus.pt",
        ),
        "facebook/sam2-hiera-large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
    }
    config_name, checkpoint_name = model_id_to_filenames[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path, add_new_layer_weights=False):
    # if add_new_layer_weights:
    #     assert ckpt_path is not None
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        if add_new_layer_weights:

            # bbox patch embed
            sd['image_encoder.trunk.bbox_mask_patch_embed.proj.weight'] = torch.concat((
                sd['image_encoder.trunk.patch_embed.proj.weight'], 
                torch.mean(sd['image_encoder.trunk.patch_embed.proj.weight'], dim=1, keepdim=True)
            ), dim=1)
            sd['image_encoder.trunk.bbox_mask_patch_embed.proj.bias'] = sd['image_encoder.trunk.patch_embed.proj.bias']

            # matting token
            sd['sam_mask_decoder.matting_mask_tokens.weight'] = torch.mean(sd['sam_mask_decoder.mask_tokens.weight'], dim=0, keepdim=True).repeat(model.sam_mask_decoder.matting_token_num, 1)
            
            output_hypernetworks_mlps_0_keys = [key for key in sd.keys() if 'output_hypernetworks_mlps.0' in key]
            for i in range(model.sam_mask_decoder.matting_token_num):
                for key in output_hypernetworks_mlps_0_keys:
                    target_key = key.replace('output_hypernetworks_mlps.0', 'matting_output_hypernetworks_mlps.{}'.format(i))
                    sd[target_key] = sd[key]

            output_upscaling_keys = [key for key in sd.keys() if 'output_upscaling' in key]
            for key in output_upscaling_keys:
                target_key = key.replace('output_upscaling', 'matting_output_upscaling')
                sd[target_key] = sd[key]

        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
