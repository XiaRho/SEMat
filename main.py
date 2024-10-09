#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from engine import MattingTrainer
import os
import time
from copy import deepcopy

from data.coconut_dataset import DistributedSamplerWrapper
from data.refmatte_dataset import RefMatteData
from data.p3m10k_dataset import P3MData
from data.dim_dataset import SIMTest, RW100Test, AIM500Test, AM2KTest, P3M500Test, RWP636Test, MattingTest
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from engine.hooks import EvalHook as MyEvalHook
from omegaconf import OmegaConf
from modeling.decoder.unet_detail_capture import DetailUNet, MattingDetailDecoder

# sam2
from sam2.build_sam import build_sam2

#running without warnings
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("detectron2")


def do_test(cfg, model, final_iter=False, next_iter=0):
    cfg = OmegaConf.create(cfg)
    if "evaluator" in cfg.dataloader and "test" in cfg.dataloader:
        task_final_iter_only = cfg.dataloader.get("final_iter_only", False)
        task_eval_period = cfg.dataloader.get("eval_period", 1)
        if not final_iter and (task_final_iter_only or next_iter % task_eval_period != 0):
            logger.info(
                f"Skip test set evaluation at iter {next_iter}, "
                f"since task_final_iter_only={task_final_iter_only}, "
                f"next_iter {next_iter} % task_eval_period {task_eval_period}"
                f"={next_iter % task_eval_period} != 0"
            )
        else:
            # add eval_iter to output_dir
            cfg.dataloader.evaluator[0].output_dir = os.path.join(cfg.train.output_dir, '{:06d}'.format(next_iter if not final_iter else cfg.train.max_iter))

            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )

            print_csv_format(ret)
        return ret

class InferenceRunner:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def __call__(self, final_iter=False, next_iter=0):
        return do_test(self.cfg, self.model, final_iter=final_iter, next_iter=next_iter)


def instantiate_dataloader_with_sampler(args, cfg):
    # instantiate dataset before modify the Sampler
    cfg.dataloader.train.dataset = instantiate(cfg.dataloader.train.dataset)

    assert len(cfg.dataloader.train.dataset.datasets) > 1
    sample_weights = []
    # if args.dataset_sample_weight is None:
    #     dataset_weight = [1.0 for _ in range(len(cfg.dataloader.train.dataset.datasets))]
    # else:
    assert len(args.dataset_sample_weight) == len(cfg.dataloader.train.dataset.datasets)
    dataset_weight = args.dataset_sample_weight
    for i, sub_dataset in enumerate(cfg.dataloader.train.dataset.datasets):
        sample_weights.append(torch.ones(len(sub_dataset), dtype=torch.float) * len(cfg.dataloader.train.dataset) * dataset_weight[i] / len(sub_dataset))
    sample_weights = torch.concat(sample_weights, dim=0)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    if comm.get_world_size() != 1:  # DDP
        cfg.dataloader.train.sampler = DistributedSamplerWrapper(sampler)
    else:  # not DDP
        cfg.dataloader.train.sampler = sampler
    cfg.dataloader.train.shuffle = None

    return instantiate(cfg.dataloader.train)


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)

    if model.lora_rank is not None:
        # model = model.init_lora(model)
        model.init_lora()

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # train_loader = instantiate(cfg.dataloader.train)
    if args.dataset_sample_weight is not None:
        train_loader = instantiate_dataloader_with_sampler(args, cfg)
    else:
        cfg.dataloader.train.dataset = instantiate(cfg.dataloader.train.dataset)
        if comm.get_world_size() == 1:  # not DDP
            cfg.dataloader.train.sampler = None
        else:
            cfg.dataloader.train.shuffle = None
            cfg.dataloader.train.sampler = DistributedSampler(dataset=cfg.dataloader.train.dataset)
        train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = MattingTrainer(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            MyEvalHook(cfg.train.eval_period, InferenceRunner(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
    # if args.resume:
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    ##############################
    ########## Training ##########
    ##############################
    if args.bs is not None:
        cfg.dataloader.train.batch_size = args.bs
    
    if args.num_workers is not None:
        cfg.dataloader.train.num_workers = args.num_workers

    if args.iter is not None:
        cfg.train.max_iter = args.iter
        cfg.train.eval_period = int(cfg.train.max_iter * 1 / 10)
        cfg.train.checkpointer.period = int(cfg.train.max_iter * 1 / 10)
        # cfg.lr_multiplier.scheduler.milestones = [int(cfg.train.max_iter * 2 / 10)]
        cfg.lr_multiplier.scheduler.num_updates = cfg.train.max_iter
        cfg.lr_multiplier.warmup_length = 250 / cfg.train.max_iter
    if args.lr_milestones is not None:
        assert len(args.lr_milestones) == len(cfg.lr_multiplier.scheduler.milestones)
        cfg.lr_multiplier.scheduler.milestones = args.lr_milestones
    cfg.lr_multiplier.scheduler.milestones = [int(cfg.train.max_iter * scale) for scale in cfg.lr_multiplier.scheduler.milestones]

    if args.lr is not None:
        cfg.optimizer.lr = args.lr

    if args.debug:
        save_config_yaml = False
        cfg.model.vis_period = 1
        cfg.train.eval_period = 10
    else:
        save_config_yaml = True
    
    if args.tag is not None:
        tag = args.tag
    else:
        tag = cfg.train.output_dir.split('/')[-1]

    if args.init_from is not None:
        cfg.train.init_checkpoint = args.init_from

    if args.eval_only:
        save_config_yaml = False
        cfg.dataloader.evaluator[0].save_eval_results_step = 1

    if args.num_gpus == 1:
        cfg.dataloader.evaluator[0].distributed = False

    ###########################
    ########## Model ##########
    ###########################
    if args.lora_rank is not None:
        cfg.model.lora_rank = args.lora_rank
        if args.lora_alpha is not None:
            cfg.model.lora_alpha = args.lora_alpha
        else:
            cfg.model.lora_alpha = args.lora_rank
        assert args.w_dora + args.w_rslora <= 1
        if args.w_dora:
            cfg.model.w_dora = True
        if args.w_rslora:
            cfg.model.w_rslora = True

    if args.model_type is not None:
        checkpoint_path = '/root/data/my_path/Matting/sam-hq/train/pretrained_checkpoint/'
        assert '_H' in cfg.train.output_dir
        if args.model_type == 'H':
            cfg.model.sam_model.model_type = 'vit_h'
            cfg.model.sam_model.checkpoint = checkpoint_path + 'sam_hq_vit_h.pth'
            cfg.model.matting_decoder.model_type = 'vit_h'
            cfg.model.matting_decoder.checkpoint_path = checkpoint_path + 'sam_vit_h_maskdecoder.pth'
        elif args.model_type == 'L':
            cfg.model.sam_model.model_type = 'vit_l'
            cfg.model.sam_model.checkpoint = checkpoint_path + 'sam_hq_vit_l.pth'
            cfg.model.matting_decoder.model_type = 'vit_l'
            cfg.model.matting_decoder.checkpoint_path = checkpoint_path + 'sam_vit_l_maskdecoder.pth'
            cfg.train.output_dir = cfg.train.output_dir.replace('_H', '_L')
        else:
            cfg.model.sam_model.model_type = 'vit_b'
            cfg.model.sam_model.checkpoint = checkpoint_path + 'sam_hq_vit_b.pth'
            cfg.model.matting_decoder.model_type = 'vit_b'
            cfg.model.matting_decoder.checkpoint_path = checkpoint_path + 'sam_vit_b_maskdecoder.pth'
            cfg.train.output_dir = cfg.train.output_dir.replace('_H', '_B')
    
    if args.lora_on_mask_decoder:
        cfg.model.lora_on_mask_decoder = True

    if args.unet_detail_capture:
        if cfg.model.sam_model.model_type == 'vit_h':
            vit_early_feat_in = 1280
        elif cfg.model.sam_model.model_type == 'vit_l':
            vit_early_feat_in = 1024
        else:
            vit_early_feat_in = 768

        cfg.model.matting_decoder.detail_capture = L(DetailUNet)(
            vit_early_feat_in = vit_early_feat_in,
            norm = torch.nn.BatchNorm2d,
        )
    
    if args.detail_matting_decoder:
        if cfg.model.sam_model.model_type == 'vit_h':
            vit_intern_feat_in = 1280
        elif cfg.model.sam_model.model_type == 'vit_l':
            vit_intern_feat_in = 1024
        else:
            vit_intern_feat_in = 768
        cfg.model.matting_decoder = L(MattingDetailDecoder)(
            vit_intern_feat_in = vit_intern_feat_in,
            vit_intern_feat_index = args.dmd_vit_f_idx
        )

    if args.finetune_all:
        assert args.lora_rank is None
        cfg.model.sam_model.mode = 'train'
        cfg.model.finetune_all = True
    
    if args.norm_mask_logits is not None:
        if args.norm_mask_logits == 0:
            cfg.model.matting_decoder.norm_mask_logits = True
        elif args.norm_mask_logits == -1:
            cfg.model.matting_decoder.norm_mask_logits = 'BN'
        elif args.norm_mask_logits == -2:
            cfg.model.matting_decoder.norm_mask_logits = 'Sigmoid'
        else:
            assert args.norm_mask_logits > 0
            cfg.model.matting_decoder.norm_mask_logits = args.norm_mask_logits
    
    if args.decoder_norm_type is not None:
        cfg.model.matting_decoder.norm_type = args.decoder_norm_type
        if hasattr(cfg.model.matting_decoder, 'detail_capture'):
            cfg.model.matting_decoder.detail_capture.norm_type = args.decoder_norm_type

    if args.frozen_sam_hq_reg is not None:
        cfg.model.frozen_sam_hq_reg = args.frozen_sam_hq_reg
        if args.reg_margin is not None:
            cfg.model.reg_margin = args.reg_margin

    if args.also_concat_trimap:
        if hasattr(cfg.model.matting_decoder, 'detail_capture'):
            cfg.model.matting_decoder.detail_capture.img_feat_in = 5
        else:
            cfg.model.matting_decoder.img_feat_in = 5
        cfg.model.matting_decoder.with_trimap = True
    
    if args.wo_hq_token_only:
        cfg.model.hq_token_only = False

    if args.w_attention_mask:
        cfg.model.w_attention_mask = True

    if args.alpha_reg_range is not None:
        cfg.model.alpha_reg_range = args.alpha_reg_range
        cfg.model.alpha_reg_weight = args.alpha_reg_weight

    if args.coconut_pl:
        cfg.model.coconut_pl = True
        cfg.model.coconut_pl_alpha = args.coconut_pl_alpha

    if args.coconut_self_training:
        cfg.model.coconut_self_training = True

    if args.backbone_condition:
        cfg.model.backbone_condition = True
        if args.condition_wo_conv:
            cfg.model.condition_wo_conv = True
        if args.w_only_bbox_cond:
            cfg.model.w_only_bbox_cond = True

    if args.coconut_only_known_l1:
        cfg.model.coconut_only_known_l1 = True

    if args.backbone_bbox_prompt is not None:
        cfg.model.backbone_bbox_prompt = args.backbone_bbox_prompt
        cfg.model.backbone_bbox_prompt_loc = args.backbone_bbox_prompt_loc
        cfg.model.backbone_bbox_prompt_loss_weight = args.backbone_bbox_prompt_loss_weight

    if args.concat_gen_trimap:
        assert args.backbone_bbox_prompt is not None
        cfg.model.matting_decoder.concat_gen_trimap = True
        cfg.model.matting_decoder.img_feat_in = 5

    if args.decoder_skip_connect is not None:
        cfg.model.matting_decoder.skip_connect = args.decoder_skip_connect

    if args.wo_hq_features:
        cfg.model.matting_decoder.img_feat_in = 3
        cfg.model.matting_decoder.wo_hq_features = True

    if args.w_all_logits:
        cfg.model.w_all_logits = True
        cfg.model.matting_decoder.img_feat_in = 8

    if args.multi_matting_decoder is not None:
        assert 2 <= args.multi_matting_decoder <= 4 and args.dmd_vit_f_idx == [0, 1, 2, 3]
        cfg.model.multi_matting_decoder = args.multi_matting_decoder
        cfg.model.matting_decoder = {
            'matting_decoder_0': cfg.model.matting_decoder
        }
        for i in range(1, args.multi_matting_decoder):
            ori_matting_decoder_cfg = deepcopy(cfg.model.matting_decoder['matting_decoder_0'])
            ori_matting_decoder_cfg.vit_intern_feat_index = ori_matting_decoder_cfg.vit_intern_feat_index[:-i]
            cfg.model.matting_decoder['matting_decoder_{}'.format(i)] = instantiate(ori_matting_decoder_cfg)
        cfg.model.matting_decoder['matting_decoder_0'] = instantiate(cfg.model.matting_decoder['matting_decoder_0'])
        cfg.model.matting_decoder = torch.nn.ModuleDict(modules = cfg.model.matting_decoder)

    if args.bbox_prompt_all_block is not None:
        cfg.model.bbox_prompt_all_block = args.bbox_prompt_all_block

    if args.matting_token:
        cfg.model.matting_token = True
        if args.matting_token_num is not None:
            cfg.model.sam_model.matting_token = args.matting_token_num
            cfg.model.matting_decoder.img_feat_in = 3 + args.matting_token_num
        else:
            cfg.model.sam_model.matting_token = 1
        if args.test_w_hq_token:
            cfg.model.test_w_hq_token = True

    if args.sam_hq_token_reg is not None:
        cfg.model.sam_model.frozen_decoder = True
        cfg.model.sam_hq_token_reg = args.sam_hq_token_reg
        if args.reg_on_sam_logits:
            cfg.model.reg_on_sam_logits = True
        if args.reg_w_bce_loss:
            cfg.model.reg_w_bce_loss = True

    if args.feat_cross_attn_fusion:
        cfg.model.feat_cross_attn_fusion = args.feat_cross_attn_fusion

    if args.wo_hq:
        cfg.model.sam_model.wo_hq = True

    if args.mask_matting_no_res_add:
        assert args.wo_hq
        cfg.model.sam_model.mask_matting_res_add = False

    if args.decoder_block_num is not None:
        cfg.model.matting_decoder.block_num = args.decoder_block_num
        if args.wo_big_kernel:
            cfg.model.matting_decoder.wo_big_kernel = True

    if args.trimap_loss_type is not None:
        cfg.model.trimap_loss_type = args.trimap_loss_type

    if args.no_multimask_output:
        cfg.model.multimask_output = False

    if args.complex_trimap_pred_layer:
        cfg.model.complex_trimap_pred_layer = True
    
    if args.matting_token_sup is not None:
        cfg.model.matting_token_sup = args.matting_token_sup
        cfg.model.matting_token_sup_loss_weight = args.matting_token_sup_loss_weight

    if args.sam2:
        cfg.model.sam2 = True
        cfg.model.sam_model = L(build_sam2)(
            config_file = 'sam2_hiera_l.yaml',
            ckpt_path = '/root/data/my_path/Matting/segment-anything-2/checkpoints/sam2_hiera_large.pt',
            device = "cuda",
            bbox_mask_matting_token = True,
            mode="train"
        )
        cfg.model.matting_decoder.sam2_multi_scale_feates = True
        if args.sam2_matting_logits_res_add:
            cfg.model.sam_model.matting_logits_res_add = True
        if args.sam2_upscaled_embedding_no_res_add:
            cfg.model.sam_model.upscaled_embedding_res_add = False

    #############################
    ########## Dataset ##########
    #############################
    if args.add_multi_fg:
        for i in range(len(cfg.dataloader.train.dataset.datasets)):
            if hasattr(cfg.dataloader.train.dataset.datasets[i], 'remove_multi_fg'):
                cfg.dataloader.train.dataset.datasets[i].remove_multi_fg = False

    if args.remove_coco_transparent:
        cfg.dataloader.train.dataset.datasets[-1].remove_coco_transparent = True

    if args.eval_w_sam_hq_mask:
        cfg.model.eval_w_sam_hq_mask = True
        cfg.dataloader.evaluator[0].eval_w_sam_hq_mask = True

    if args.eval_dataset == 'AIM500':
        cfg.dataloader.test.dataset.datasets = [cfg.dataloader.test.dataset.datasets[0]]
        cfg.dataloader.evaluator[0].eval_dataset_type = ['AIM500']
    elif args.eval_dataset == 'RW100':
        cfg.dataloader.test.dataset.datasets = [cfg.dataloader.test.dataset.datasets[1]]
        cfg.dataloader.evaluator[0].eval_dataset_type = ['RW100']
    elif args.eval_dataset == 'AM2K':
        cfg.dataloader.test.dataset.datasets = [cfg.dataloader.test.dataset.datasets[2]]
        cfg.dataloader.evaluator[0].eval_dataset_type = ['AM2K']
    elif args.eval_dataset == 'P3M500':
        cfg.dataloader.test.dataset.datasets = [cfg.dataloader.test.dataset.datasets[3]]
        cfg.dataloader.evaluator[0].eval_dataset_type = ['P3M500']
    elif args.eval_dataset == 'RWP636':
        cfg.dataloader.test.dataset.datasets = [cfg.dataloader.test.dataset.datasets[4]]
        cfg.dataloader.evaluator[0].eval_dataset_type = ['RWP636']
    elif args.eval_dataset == 'SIM':
        cfg.dataloader.test.dataset.datasets = [cfg.dataloader.test.dataset.datasets[5]]
        cfg.dataloader.evaluator[0].eval_dataset_type = ['SIM']

    if args.random_crop_bbox is not None:
        for i in range(len(cfg.dataloader.train.dataset.datasets)):
            if hasattr(cfg.dataloader.train.dataset.datasets[i], 'random_crop_bbox'):
                cfg.dataloader.train.dataset.datasets[i].random_crop_bbox = args.random_crop_bbox

    if args.key_sample_ratio is not None:
        cfg.dataloader.train.dataset.datasets[0].data.key_sample_ratio = args.key_sample_ratio
    
    if args.coconut_num_ratio is not None:
        cfg.dataloader.train.dataset.datasets[-1].coconut_num_ratio = args.coconut_num_ratio
    
    if args.alpha_less_02:
        cfg.dataloader.train.dataset.datasets[0].data.alpha_ratio_json[0] = cfg.dataloader.train.dataset.datasets[0].data.alpha_ratio_json[0].replace('less_08', 'less_02')
        cfg.dataloader.train.dataset.datasets[0].data.alpha_ratio_json[2] = cfg.dataloader.train.dataset.datasets[0].data.alpha_ratio_json[2].replace('less_08', 'less_02')

    if args.coconut_0730_new_MF_accessory:
        cfg.dataloader.train.dataset.datasets[-1].miou_json = ''
        cfg.dataloader.train.dataset.datasets[-1].json_path = '/root/data/my_path/Matting/DiffMatte-main/24-07-30_coco-nut_accessory_new_MF_matting.json'

    if args.no_alpha_select:
        for i in range(len(cfg.dataloader.train.dataset.datasets[0].data.alpha_ratio_json)):
            cfg.dataloader.train.dataset.datasets[0].data.alpha_ratio_json[i] = None

    if args.fg_have_bg_num is not None:
        assert len(args.fg_have_bg_num) == len(cfg.dataloader.train.dataset.datasets[0].data.alpha_dir)
        cfg.dataloader.train.dataset.datasets[0].data.fg_have_bg_num = args.fg_have_bg_num

    if args.random_auto_matting is not None:
        cfg.dataloader.train.dataset.datasets[0].random_auto_matting = args.random_auto_matting

    if args.bbox_offset_factor is not None:
        for i in range(len(cfg.dataloader.train.dataset.datasets)):
            if hasattr(cfg.dataloader.train.dataset.datasets[i], 'bbox_offset_factor'):
                cfg.dataloader.train.dataset.datasets[i].bbox_offset_factor = args.bbox_offset_factor

    if args.wo_accessory_fusion:
        cfg.dataloader.train.dataset.datasets[-1].wo_accessory_fusion = True
    
    if args.wo_mask_to_mattes:
        cfg.dataloader.train.dataset.datasets[-1].wo_mask_to_mattes = True

    if args.wo_coco_nut:
        cfg.dataloader.train.dataset.datasets = [cfg.dataloader.train.dataset.datasets[0]]
    
    if args.replace_coconut_with_refmatte:
        cfg.dataloader.train.dataset.datasets[-1] = L(RefMatteData)(
            data_root_path='/root/data/my_path_b/public_data/data/matting/RefMatte/RefMatte/train/img'
        )

    if args.replace_coconut_with_p3m10k is not None:
        cfg.dataloader.train.dataset.datasets[-1] = L(P3MData)(
            data_root_path = '/root/data/my_path_b/public_data/data/matting/P3M-10k/train/blurred_image/',
            num_ratio = args.replace_coconut_with_p3m10k
        )

    # add time to save_path
    now_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    now_time = now_time[2:8] + '-' + now_time[8:]
    new_tag = now_time + '_' + tag
    new_tag = 'DEBUG_' + new_tag if args.debug else new_tag    
    ori_tag = cfg.train.output_dir.split('/')[-1]
    cfg.train.output_dir = cfg.train.output_dir.replace(ori_tag, new_tag)
    cfg.model.output_dir = cfg.train.output_dir

    try:
        default_setup(cfg, args, save_config_yaml=save_config_yaml)
    except TypeError:
        default_setup(cfg, args)

    logger.info(f"Running with config:\n{LazyConfig.to_py(cfg)}")

    if args.eval_only:
        if hasattr(cfg.model.sam_model, 'ckpt_path'):
            cfg.model.sam_model.ckpt_path = None
        else:
            cfg.model.sam_model.checkpoint = None
        model = instantiate(cfg.model)
        if model.lora_rank is not None:
            # model = model.init_lora(model)
            model.init_lora()
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":

    default_args = default_argument_parser()
    default_args.add_argument("--bs", type=int, default=None)
    default_args.add_argument("--num_workers", type=int, default=None)
    default_args.add_argument("--iter", type=int, default=None)
    default_args.add_argument("--lr", type=float, default=None)
    default_args.add_argument("--debug", action="store_true")
    default_args.add_argument("--lora_rank", type=int, default=None)
    default_args.add_argument("--lora_alpha", type=int, default=None)
    default_args.add_argument("--w_dora", action="store_true")
    default_args.add_argument("--w_rslora", action="store_true")
    default_args.add_argument("--tag", type=str, default=None)
    default_args.add_argument("--dataset_sample_weight", type=float, default=None, nargs="*")
    default_args.add_argument("--init_from", type=str, help="init from the given checkpoint", default=None)
    default_args.add_argument("--model_type", type=str, default=None, choices=[None, 'H', 'L', 'B'])
    default_args.add_argument("--lora_on_mask_decoder", action="store_true")
    default_args.add_argument("--unet_detail_capture", action="store_true")
    default_args.add_argument("--finetune_all", action="store_true")
    default_args.add_argument("--norm_mask_logits", type=float, default=None)
    default_args.add_argument("--decoder_norm_type", type=str, default=None, choices=[None, 'BN', 'LN', 'SyncBN'])
    default_args.add_argument("--frozen_sam_hq_reg", type=float, default=None)
    default_args.add_argument("--reg_margin", type=float, default=None)
    default_args.add_argument("--also_concat_trimap", action="store_true")
    default_args.add_argument("--wo_hq_token_only", action="store_true")
    default_args.add_argument("--add_multi_fg", action="store_true")
    default_args.add_argument("--remove_coco_transparent", action="store_true")
    default_args.add_argument("--eval_dataset", type=str, choices=['SIM', 'RW100', 'AIM500', 'AM2K', 'P3M500', 'RWP636'])
    default_args.add_argument("--w_attention_mask", action="store_true")
    default_args.add_argument("--random_crop_bbox", type=float, default=None)
    default_args.add_argument("--alpha_reg_range", type=float, default=None)
    default_args.add_argument("--alpha_reg_weight", type=float, default=1.0)
    default_args.add_argument("--coconut_pl", action="store_true")
    default_args.add_argument("--coconut_pl_alpha", type=float, default=1.0)
    default_args.add_argument("--coconut_self_training", action="store_true")
    default_args.add_argument("--eval_w_sam_hq_mask", action="store_true")
    default_args.add_argument("--key_sample_ratio", type=float, default=None)
    default_args.add_argument("--coconut_num_ratio", type=float, default=None)
    default_args.add_argument("--alpha_less_02", action="store_true")
    default_args.add_argument("--coco_miou_0723", action="store_true")
    default_args.add_argument("--backbone_condition", action="store_true")
    default_args.add_argument("--condition_wo_conv", action="store_true")
    default_args.add_argument("--w_only_bbox_cond", action="store_true")
    default_args.add_argument("--coconut_only_known_l1", action="store_true")
    default_args.add_argument("--backbone_bbox_prompt", type=str, choices=['bbox', 'trimap', 'alpha', 'alpha_trimap', None], default=None)
    default_args.add_argument("--backbone_bbox_prompt_loc", type=int, default=[2, 3], nargs="*")
    default_args.add_argument("--backbone_bbox_prompt_loss_weight", type=float, default=1.0)
    default_args.add_argument("--coconut_0730_new_MF_accessory", action="store_true")
    default_args.add_argument("--detail_matting_decoder", action="store_true")
    default_args.add_argument("--dmd_vit_f_idx", type=int, default=[0, 1, 2, 3], nargs="*")
    default_args.add_argument("--concat_gen_trimap", action="store_true")
    default_args.add_argument("--decoder_skip_connect", type=str, choices=['sum', 'concat', None], default=None)
    default_args.add_argument("--no_alpha_select", action="store_true")
    default_args.add_argument("--wo_hq_features", action="store_true")
    default_args.add_argument("--w_all_logits", action="store_true")
    default_args.add_argument("--bbox_prompt_all_block", type=str, default=None)
    default_args.add_argument("--multi_matting_decoder", type=int, default=None)
    default_args.add_argument("--matting_token", action="store_true")
    default_args.add_argument("--test_w_hq_token", action="store_true")
    default_args.add_argument("--sam_hq_token_reg", type=float, default=None)
    default_args.add_argument("--feat_cross_attn_fusion", action="store_true")
    default_args.add_argument("--wo_hq", action="store_true")
    default_args.add_argument("--decoder_block_num", type=int, default=None)
    default_args.add_argument("--trimap_loss_type", type=str, choices=['CE', 'F1', 'F2', 'NF1', 'NF2', 'NF3', 'GHM', 'NGHM', None], default=None)
    default_args.add_argument("--reg_on_sam_logits", action="store_true")
    default_args.add_argument("--reg_w_bce_loss", action="store_true")
    default_args.add_argument("--no_multimask_output", action="store_true")
    default_args.add_argument("--fg_have_bg_num", type=int, default=None, nargs="*")
    default_args.add_argument("--wo_coco_nut", action="store_true")
    default_args.add_argument("--random_auto_matting", type=float, default=None)
    default_args.add_argument("--bbox_offset_factor", type=float, default=None)
    default_args.add_argument("--wo_big_kernel", action="store_true")
    default_args.add_argument("--lr_milestones", type=float, default=None, nargs="*")
    default_args.add_argument("--complex_trimap_pred_layer", action="store_true")
    default_args.add_argument("--wo_accessory_fusion", action="store_true")
    default_args.add_argument("--wo_mask_to_mattes", action="store_true")
    default_args.add_argument("--matting_token_sup", type=str, choices=['alpha', 'trimap', None], default=None)
    default_args.add_argument("--matting_token_sup_loss_weight", type=float, default=0.05)
    default_args.add_argument("--matting_token_num", type=int, default=None)
    default_args.add_argument("--replace_coconut_with_refmatte", action="store_true")
    default_args.add_argument("--replace_coconut_with_p3m10k", type=float, default=None)
    default_args.add_argument("--sam2", action="store_true")
    default_args.add_argument("--mask_matting_no_res_add", action="store_true")
    default_args.add_argument("--sam2_matting_logits_res_add", action="store_true")
    default_args.add_argument("--sam2_upscaled_embedding_no_res_add", action="store_true")

    args = default_args.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
