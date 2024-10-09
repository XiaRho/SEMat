from .common.train import train
from .semantic_enhanced_matting.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .semantic_enhanced_matting.dataloader import dataloader
from modeling.decoder.unet_detail_capture import MattingDetailDecoder
from detectron2.config import LazyCall as L
from sam2.build_sam import build_sam2

model.sam_model.model_type = 'vit_l'
model.sam_model.checkpoint = None
model.vis_period = 200
model.output_dir = '?'

train.max_iter = 60000
train.eval_period = int(train.max_iter * 1 / 10)
train.checkpointer.period = int(train.max_iter * 1 / 10)
train.checkpointer.max_to_keep = 1

optimizer.lr = 5e-5

lr_multiplier.scheduler.values = [1.0, 0.5, 0.2]
lr_multiplier.scheduler.milestones = [0.5, 0.75]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.output_dir = './work_dirs/SEMat_SAM2'

model.sam2 = True
model.sam_model = L(build_sam2)(
    config_file = 'sam2_hiera_l.yaml',
    ckpt_path = None,
    device = "cuda",
    bbox_mask_matting_token = True,
    mode="train",
    upscaled_embedding_res_add = False
)
model.lora_rank = 16
model.lora_alpha = 16
model.matting_decoder = L(MattingDetailDecoder)(
    vit_intern_feat_in = 1024,
    vit_intern_feat_index = [0, 1, 2, 3],
    norm_type = 'SyncBN',
    block_num = 2,
    img_feat_in = 6,
    norm_mask_logits = 6.5,
    sam2_multi_scale_feates = True
)
model.backbone_bbox_prompt = 'bbox'
model.backbone_bbox_prompt_loc = [2, 3]
model.backbone_bbox_prompt_loss_weight = 1.0
model.matting_token = True
model.sam_hq_token_reg = 0.2
model.reg_w_bce_loss = True
model.matting_token_sup = 'trimap'
model.matting_token_sup_loss_weight = 0.05
model.trimap_loss_type = 'NGHM'
