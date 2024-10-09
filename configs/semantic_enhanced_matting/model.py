from detectron2.config import LazyCall as L

from modeling import Detail_Capture, MattingCriterion
from modeling.meta_arch import SamHqMatte
from modeling.semantic_enhanced_matting.build_sam import sam_model_registry_def
# from modeling.sam_hq_matting.predictor import SamPredictor
from modeling.semantic_enhanced_matting import MaskDecoderMatting

mask_token_only = False

model = L(SamHqMatte)(

    # original sam_hq
    sam_model = L(sam_model_registry_def)(
        model_type = 'vit_b',
        checkpoint = None,
    ),
    hq_token_only = True,
    hq_features_type = 'Final',
    multimask_output = True,

    # loss function
    criterion=L(MattingCriterion)(
        losses = ['unknown_l1_loss', 'known_l1_loss', 'loss_pha_laplacian', 'loss_gradient_penalty']
    ),
    
    # other params.
    pixel_mean = [123.675 / 255., 116.280 / 255., 103.530 / 255.],
    pixel_std = [58.395 / 255., 57.120 / 255., 57.375 / 255.],

    lora_rank = None,
    lora_alpha = None,
    w_dora = False,
    w_rslora = False,
)