import os
import time
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DistributedSampler, Sampler
from torchvision import transforms
from detectron2.utils.logger import setup_logger
from typing import Optional
from operator import itemgetter
from collections import defaultdict

from data.dim_dataset import GenBBox


def random_interp():
    return np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])


class SplitConcatImage(object):

    def __init__(self, concat_num=4, wo_mask_to_mattes=False):
        self.concat_num = concat_num
        self.wo_mask_to_mattes = wo_mask_to_mattes
        if self.wo_mask_to_mattes:
            assert self.concat_num == 5

    def __call__(self, concat_image):
        if isinstance(concat_image, list):
            concat_image, image_path = concat_image[0], concat_image[1]
        else:
            image_path = None
        H, W, _ = concat_image.shape

        concat_num = self.concat_num
        if image_path is not None:
            if '06-14' in image_path:
                concat_num = 4
            elif 'ori_mask' in image_path or 'SEMat' in image_path:
                concat_num = 3
            else:
                concat_num = 5
        
        assert W % concat_num == 0
        W = W // concat_num

        image = concat_image[:H, :W]
        if self.concat_num != 3:
            trimap = concat_image[:H, (concat_num - 2) * W: (concat_num - 1) * W]
            if self.wo_mask_to_mattes:
                alpha = concat_image[:H, 2 * W: 3 * W]
            else:
                alpha = concat_image[:H, (concat_num - 1) * W: concat_num * W]
        else:
            trimap = concat_image[:H, (concat_num - 1) * W: concat_num * W]
            alpha = concat_image[:H, (concat_num - 2) * W: (concat_num - 1) * W]

        return {'image': image, 'trimap': trimap, 'alpha': alpha}


class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0, 1) < self.prob:
            for key in sample.keys():
                sample[key] = cv2.flip(sample[key], 1)
        return sample

class EmptyAug(object):
    def __call__(self, sample):
        return sample

class RandomReszieCrop(object):

    def __init__(self, output_size=1024, aug_scale_min=0.5, aug_scale_max=1.5):
        self.desired_size = output_size
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def __call__(self, sample):
        H, W, _ = sample['image'].shape

        if self.aug_scale_min == 1.0 and self.aug_scale_max == 1.0:
            crop_H, crop_W = H, W
            crop_y1, crop_y2 = 0, crop_H
            crop_x1, crop_x2 = 0, crop_W
            scale_W, scaled_H = W, H
        elif self.aug_scale_min == -1.0 and self.aug_scale_max == -1.0:
            scale = min(self.desired_size / H, self.desired_size / W)
            scaled_H, scale_W = round(H * scale), round(W * scale)
            crop_H, crop_W = scaled_H, scale_W
            crop_y1, crop_y2 = 0, crop_H
            crop_x1, crop_x2 = 0, crop_W
        else:
            # random size
            random_scale = np.random.uniform(0, 1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min  # random_val: 0.5 ~ 1.5
            scaled_size = round(random_scale * self.desired_size)

            scale = min(scaled_size / H, scaled_size / W)
            scaled_H, scale_W = round(H * scale), round(W * scale)

            # random crop
            crop_H, crop_W = min(self.desired_size, scaled_H), min(self.desired_size, scale_W)  # crop_size
            margin_H, margin_W = max(scaled_H - crop_H, 0), max(scale_W - crop_W, 0)
            offset_H, offset_W = np.random.randint(0, margin_H + 1), np.random.randint(0, margin_W + 1)
            crop_y1, crop_y2 = offset_H, offset_H + crop_H
            crop_x1, crop_x2 = offset_W, offset_W + crop_W

        for key in sample.keys():
            sample[key] = cv2.resize(sample[key], (scale_W, scaled_H), interpolation=random_interp())[crop_y1: crop_y2, crop_x1: crop_x2, :]  # resize and crop
            padding = np.zeros(shape=(self.desired_size, self.desired_size, 3), dtype=sample[key].dtype)  # pad to desired_size
            padding[: crop_H, : crop_W, :] = sample[key]
            sample[key] = padding

        return sample


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):

        image = sample['image']

        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        image = cv2.cvtColor(image.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        image[:, :, 0] = np.remainder(image[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = image[:, :, 1].mean()

        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = image[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        image[:, :, 1] = sat
        # Value noise
        val_bar = image[:, :, 2].mean()

        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = image[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        image[:, :, 2] = val
        # convert back to BGR space
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        sample['image'] = image * 255

        return sample


class ToTensor(object):

    def __call__(self, sample):
        image, alpha, trimap = sample['image'][:, :, ::-1], sample['alpha'], sample['trimap']

        # image
        image = image.transpose((2, 0, 1)) / 255.
        sample['image'] = torch.from_numpy(image).float()

        # alpha
        alpha = alpha.transpose((2, 0, 1))[0: 1] / 255.
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        sample['alpha'] = torch.from_numpy(alpha).float()

        # trimap
        trimap = trimap.transpose((2, 0, 1))[0: 1] / 1.
        sample['trimap'] = torch.from_numpy(trimap).float()
        sample['trimap'][sample['trimap'] < 85] = 0
        sample['trimap'][sample['trimap'] >= 170] = 1
        sample['trimap'][sample['trimap'] >= 85] = 0.5

        return sample
    

class COCONutData(Dataset):
    def __init__(
        self, 
        json_path, 
        data_root_path, 
        output_size = 512, 
        aug_scale_min = 0.5, 
        aug_scale_max = 1.5,
        with_bbox = False, 
        bbox_offset_factor = None,
        phase = "train",
        min_miou = 95,
        miou_json = '',
        remove_coco_transparent = False,
        coconut_num_ratio = None,
        return_multi_fg_info = False,
        wo_accessory_fusion = False,
        wo_mask_to_mattes = False,
        return_image_name = False,
    ):
        
        self.data_root_path = data_root_path
        self.output_size = output_size
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.with_bbox = with_bbox
        self.bbox_offset_factor = bbox_offset_factor
        self.phase = phase
        self.min_miou = min_miou
        self.miou_json = miou_json
        self.remove_coco_transparent = remove_coco_transparent
        self.coconut_num_ratio = coconut_num_ratio
        self.return_multi_fg_info = return_multi_fg_info
        self.wo_accessory_fusion = wo_accessory_fusion # TODO
        self.wo_mask_to_mattes = wo_mask_to_mattes
        self.return_image_name = return_image_name
        assert self.wo_accessory_fusion + self.wo_mask_to_mattes <= 1
        assert self.phase == 'train'

        self.data_path = []
        with open(json_path, "r") as file:
            coconut_matting_info = json.load(file)
        
        if self.miou_json != '':
            name_2_miou_dict = defaultdict(int)
            with open(self.miou_json, "r") as file:
                coconut_matting_miou = json.load(file)
            for miou, name in coconut_matting_miou:
                name_2_miou_dict[name] = miou
            for i in coconut_matting_info:
                if 'accessory' in i['save_path']:
                    self.data_path.append(i['save_path'])
                elif name_2_miou_dict[i['save_path'].split('/')[-1]] >= self.min_miou:
                    if not (self.remove_coco_transparent and 'glass' in i['save_path']):
                        self.data_path.append(i['save_path'])
        else:
            for i in coconut_matting_info:
                self.data_path.append(i['save_path'])

        if 'accessory' in json_path:
            concat_num = 5
        elif 'ori_mask' in json_path:
            concat_num = 3
        else:
            concat_num = 4

        train_trans = [
            SplitConcatImage(concat_num, wo_mask_to_mattes = self.wo_mask_to_mattes),
            RandomHorizontalFlip(prob=0 if hasattr(self, 'return_image_name') and self.return_image_name else 0.5),
            RandomReszieCrop(self.output_size, self.aug_scale_min, self.aug_scale_max),
            EmptyAug() if hasattr(self, 'return_image_name') and self.return_image_name else RandomJitter(),
            ToTensor(),
            GenBBox(bbox_offset_factor=self.bbox_offset_factor)
        ]
        self.transform = transforms.Compose(train_trans)
        print('coconut num: ', len(self.data_path) * self.coconut_num_ratio if self.coconut_num_ratio is not None else len(self.data_path))

    def __getitem__(self, idx):
        if self.coconut_num_ratio is not None:
            if self.coconut_num_ratio < 1.0 or idx >= len(self.data_path):
                idx = np.random.randint(0, len(self.data_path))
        concat_image = cv2.imread(os.path.join(self.data_root_path, self.data_path[idx]))
        sample = self.transform([concat_image, self.data_path[idx]])
        sample['dataset_name'] = 'COCONut'
        if self.return_multi_fg_info:
            sample['multi_fg'] = False
        if hasattr(self, 'return_image_name') and self.return_image_name:
            sample['image_name'] = self.data_path[idx]
        return sample

    def __len__(self):
        if self.coconut_num_ratio is not None:
            return int(len(self.data_path) * self.coconut_num_ratio)
        else:
            return len(self.data_path)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)
    

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
    

if __name__ == '__main__':

    

    dataset = COCONutData(
        json_path = '/root/data/my_path/Matting/DiffMatte-main/24-06-14_coco-nut_matting.json', 
        data_root_path = '/root/data/my_path/Matting/DiffMatte-main', 
        output_size = 1024, 
        aug_scale_min = 0.5, 
        aug_scale_max = 1.5,
        with_bbox = True, 
        bbox_offset_factor = 0.1,
        phase = "train"
    )
    data = dataset[0]

    for key, val in data.items():
        print(key, val.shape, torch.min(val), torch.max(val))