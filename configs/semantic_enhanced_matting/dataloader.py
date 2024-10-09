from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset
from detectron2.config import LazyCall as L

from data.dim_dataset import build_d2_test_dataloader, AdobeCompositionEvaluator, adobe_composition_collate_fn, RW100Test, AIM500Test, AM2KTest, P3M500Test, RWP636Test, SIMTest

AIM500_PATH = '/path/to/datasets/AIM-500'
RW100_PATH = '/path/to/datasets/RefMatte_RW_100'
AM2K_PATH = '/path/to/datasets/AM-2K'
P3M500_PATH = '/path/to/datasets/P3M-10k/validation/P3M-500-NP'
RWP636_PATH = '/path/to/datasets/RealWorldPortrait-636'
SIM_PATH = '/path/to/datasets/SIMD/generated_testset'

dataloader = OmegaConf.create()
test_dataset = L(ConcatDataset)(
    datasets = [
        L(AIM500Test)(
            data_dir = AIM500_PATH,
            target_size = 1024,
            multi_fg = True,
        ),
        L(RW100Test)(
            data_dir = RW100_PATH,
            target_size = 1024,
            multi_fg = True,
        ),
        L(AM2KTest)(
            data_dir = AM2K_PATH,
            target_size = 1024,
            multi_fg = True,
        ),
        L(P3M500Test)(
            data_dir = P3M500_PATH,
            target_size = 1024,
            multi_fg = True,
        ),
        L(RWP636Test)(
            data_dir = RWP636_PATH,
            target_size = 1024,
            multi_fg = True
        ),
        L(SIMTest)(
            data_dir = SIM_PATH,
            target_size = 1024,
            multi_fg = True
        )
    ]
)

dataloader.test = L(build_d2_test_dataloader)(
    dataset = test_dataset,
    local_batch_size = 1,
    num_workers = 4,
    collate_fn = adobe_composition_collate_fn
)

dataloader.evaluator = L(AdobeCompositionEvaluator)(
    save_eval_results_step = 10, 
    output_dir = None,  # modify in EvalHook (do_test)
    eval_dataset_type = ['RW100', 'AIM500', 'AM2K', 'P3M500', 'RWP636', 'SIM'],
    distributed = True,
),
