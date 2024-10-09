import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import math
import torch.nn.functional as F


class GenBBox(object):
    def __init__(self, bbox_offset_factor = 0.1, random_crop_bbox = None, train_or_test = 'train', dataset_type = None, random_auto_matting=None):
        self.bbox_offset_factor = bbox_offset_factor
        self.random_crop_bbox = random_crop_bbox
        self.train_or_test = train_or_test
        self.dataset_type = dataset_type
        self.random_auto_matting = random_auto_matting

    def __call__(self, sample):

        alpha = sample['alpha']  # [1, H, W] 0.0 ~ 1.0
        indices = torch.nonzero(alpha[0], as_tuple=True)

        if len(indices[0]) > 0:

            min_x, min_y = torch.min(indices[1]), torch.min(indices[0])
            max_x, max_y = torch.max(indices[1]), torch.max(indices[0])

            if self.random_crop_bbox is not None and np.random.uniform(0, 1) < self.random_crop_bbox:
                ori_h_w = (sample['alpha'].shape[-2], sample['alpha'].shape[-1])
                sample['alpha'] = F.interpolate(sample['alpha'][None, :, min_y: max_y + 1, min_x: max_x + 1], size=ori_h_w, mode='bilinear', align_corners=False)[0]
                sample['image'] = F.interpolate(sample['image'][None, :, min_y: max_y + 1, min_x: max_x + 1], size=ori_h_w, mode='bilinear', align_corners=False)[0]
                sample['trimap'] = F.interpolate(sample['trimap'][None, :, min_y: max_y + 1, min_x: max_x + 1], size=ori_h_w, mode='nearest')[0]
                bbox = torch.tensor([[0, 0, ori_h_w[1] - 1, ori_h_w[0] - 1]])

            elif self.bbox_offset_factor != 0:
                bbox_w = max(1, max_x - min_x)
                bbox_h = max(1, max_y - min_y)
                offset_w = math.ceil(self.bbox_offset_factor * bbox_w)
                offset_h = math.ceil(self.bbox_offset_factor * bbox_h)

                min_x = max(0, min_x + np.random.randint(-offset_w, offset_w))
                max_x = min(alpha.shape[2] - 1, max_x + np.random.randint(-offset_w, offset_w))
                min_y = max(0, min_y + np.random.randint(-offset_h, offset_h))
                max_y = min(alpha.shape[1] - 1, max_y + np.random.randint(-offset_h, offset_h))
                bbox = torch.tensor([[min_x, min_y, max_x, max_y]])
            else:
                bbox = torch.tensor([[min_x, min_y, max_x, max_y]])
            
            if self.random_auto_matting is not None and np.random.uniform(0, 1) < self.random_auto_matting:
                bbox = torch.tensor([[0, 0, alpha.shape[2] - 1, alpha.shape[1] - 1]])

        else:
            bbox = torch.zeros(1, 4)

        sample['bbox'] = bbox.float()
        return sample

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
        sample['trimap'] = sample['trimap'][:, :, None].repeat(3, axis=-1)
        sample['alpha'] = sample['alpha'][:, :, None].repeat(3, axis=-1)

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
    

class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    def __call__(self, sample):
        alpha = sample['alpha']
        h, w = alpha.shape

        max_kernel_size = max(30, int((min(h,w) / 2048) * 30))

        ### generate trimap
        fg_mask = (alpha / 255.0 + 1e-5).astype(np.int32).astype(np.uint8)
        bg_mask = (1 - alpha / 255.0 + 1e-5).astype(np.int32).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        return sample
    

class P3MData(Dataset):
    def __init__(
        self, 
        data_root_path = '/root/data/my_path_b/public_data/data/matting/P3M-10k/train/blurred_image/', 
        output_size = 1024, 
        aug_scale_min = 0.8, 
        aug_scale_max = 1.5,
        with_bbox = True, 
        bbox_offset_factor = 0.05,
        num_ratio = 4.06,  # 9421 * 4.06 = 38249.26 (38251)
    ):
        
        self.data_root_path = data_root_path
        self.output_size = output_size
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.with_bbox = with_bbox
        self.bbox_offset_factor = bbox_offset_factor
        self.num_ratio = num_ratio

        self.image_names = os.listdir(self.data_root_path)
        self.image_names = [i for i in self.image_names if 'jpg' in i]
        self.image_names.sort()

        train_trans = [
            RandomHorizontalFlip(prob=0 if hasattr(self, 'return_image_name') and self.return_image_name else 0.5),
            GenTrimap(),
            RandomReszieCrop(self.output_size, self.aug_scale_min, self.aug_scale_max),
            RandomJitter(),
            ToTensor(),
            GenBBox(bbox_offset_factor=self.bbox_offset_factor)
        ]
        self.transform = transforms.Compose(train_trans)

    def __getitem__(self, idx):

        if self.num_ratio is not None:
            if self.num_ratio < 1.0:
                idx = np.random.randint(0, len(self.image_names))
            else:
                idx = idx % len(self.image_names)

        image_path = os.path.join(self.data_root_path, self.image_names[idx])
        alpha_path = image_path.replace('jpg', 'png').replace('blurred_image', 'mask')

        sample = self.transform({
            'image': cv2.imread(image_path),
            'alpha': cv2.imread(alpha_path, 0),
        })

        sample['dataset_name'] = 'P3M'
        sample['multi_fg'] = False

        return sample

    def __len__(self):
        if self.num_ratio is not None:
            return int(len(self.image_names) * self.num_ratio)
        else:
            return len(self.image_names)


if __name__ == '__main__':

    dataset = P3MData()
    data = dataset[0]
    print(len(dataset))
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape, torch.min(val), torch.max(val), torch.unique(val))
        else:
            print(key, val)