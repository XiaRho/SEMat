'''
Dataloader to process Adobe Image Matting Dataset.

From GCA_Matting(https://github.com/Yaoyi-Li/GCA-Matting/tree/master/dataloader)
'''
import os
import glob
import logging
import os.path as osp
import functools
import numpy as np
import torch
import cv2
import math
import numbers
import random
import pickle
from   torch.utils.data import Dataset, DataLoader
from   torch.nn import functional as F
from   torchvision import transforms
from easydict import EasyDict
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.data import build_detection_test_loader
import torchvision.transforms.functional

import json
from PIL import Image
from detectron2.evaluation.evaluator import DatasetEvaluator
from collections import defaultdict

from data.evaluate import compute_sad_loss, compute_mse_loss, compute_mad_loss, compute_gradient_loss, compute_connectivity_error

# Base default config
CONFIG = EasyDict({})

# Model config
CONFIG.model = EasyDict({})
# one-hot or class, choice: [3, 1]
CONFIG.model.trimap_channel = 1

# Dataloader config
CONFIG.data = EasyDict({})
# feed forward image size (untested)
CONFIG.data.crop_size = 512
# composition of two foregrounds, affine transform, crop and HSV jitter
CONFIG.data.cutmask_prob = 0.25
CONFIG.data.augmentation = True
CONFIG.data.random_interp = True

class Prefetcher():
    """
    Modified from the data_prefetcher in https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """
    def __init__(self, loader):
        self.orig_loader = loader
        self.stream = torch.cuda.Stream()
        self.next_sample = None

    def preload(self):
        try:
            self.next_sample = next(self.loader)
        except StopIteration:
            self.next_sample = None
            return

        with torch.cuda.stream(self.stream):
            for key, value in self.next_sample.items():
                if isinstance(value, torch.Tensor):
                    self.next_sample[key] = value.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        sample = self.next_sample
        if sample is not None:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key].record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            # throw stop exception if there is no more data to perform as a default dataloader
            raise StopIteration("No samples in loader. example: `iterator = iter(Prefetcher(loader)); "
                                "data = next(iterator)`")
        return sample

    def __iter__(self):
        self.loader = iter(self.orig_loader)
        self.preload()
        return self


class ImageFile(object):
    def __init__(self, phase='train'):
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        name_sets = [self._get_name_set(d) for d in dirs]

        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]

class ImageFileTrain(ImageFile):
    def __init__(
        self,
        alpha_dir="train_alpha",
        fg_dir="train_fg",
        bg_dir="train_bg",
        alpha_ext=".jpg",
        fg_ext=".jpg",
        bg_ext=".jpg",
        fg_have_bg_num=None,
        alpha_ratio_json = None,
        alpha_min_ratio = None,
        key_sample_ratio = None,
    ):
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir  = alpha_dir
        self.fg_dir     = fg_dir
        self.bg_dir     = bg_dir
        self.alpha_ext  = alpha_ext
        self.fg_ext     = fg_ext
        self.bg_ext     = bg_ext
        logger = setup_logger(name=__name__)

        if not isinstance(self.alpha_dir, str):
            assert len(self.alpha_dir) == len(self.fg_dir) == len(alpha_ext) == len(fg_ext)
            self.valid_fg_list = []
            self.alpha = []
            self.fg = []
            self.key_alpha = []
            self.key_fg = []
            for i in range(len(self.alpha_dir)):
                valid_fg_list = self._get_valid_names(self.fg_dir[i], self.alpha_dir[i])
                valid_fg_list.sort()
                alpha = self._list_abspath(self.alpha_dir[i], self.alpha_ext[i], valid_fg_list)
                fg = self._list_abspath(self.fg_dir[i], self.fg_ext[i], valid_fg_list)
                self.valid_fg_list += valid_fg_list

                self.alpha += alpha * fg_have_bg_num[i]
                self.fg += fg * fg_have_bg_num[i]

                if alpha_ratio_json[i] is not None:
                    tmp_key_alpha = []
                    tmp_key_fg = []
                    name_to_alpha_path = dict()
                    for name in alpha:
                        name_to_alpha_path[name.split('/')[-1].split('.')[0]] = name
                    name_to_fg_path = dict()
                    for name in fg:
                        name_to_fg_path[name.split('/')[-1].split('.')[0]] = name

                    with open(alpha_ratio_json[i], 'r') as file:
                        alpha_ratio_list = json.load(file)
                    for ratio, name in alpha_ratio_list:
                        if ratio < alpha_min_ratio[i]:
                            break
                        tmp_key_alpha.append(name_to_alpha_path[name.split('.')[0]])
                        tmp_key_fg.append(name_to_fg_path[name.split('.')[0]])

                    self.key_alpha.extend(tmp_key_alpha * fg_have_bg_num[i])
                    self.key_fg.extend(tmp_key_fg * fg_have_bg_num[i])

            if len(self.key_alpha) != 0 and key_sample_ratio > 0:
                repeat_num = key_sample_ratio * (len(self.alpha) - len(self.key_alpha)) / len(self.key_alpha) / (1 - key_sample_ratio) - 1
                print('key sample num:', len(self.key_alpha), ', repeat num: ', repeat_num)
                for i in range(math.ceil(repeat_num)):
                    self.alpha += self.key_alpha
                    self.fg += self.key_fg

        else:
            self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir)
            self.valid_fg_list.sort()
            self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)
            self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
            
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]
        self.valid_bg_list.sort()

        if fg_have_bg_num is not None:
            # assert fg_have_bg_num * len(self.valid_fg_list) <= len(self.valid_bg_list)
            # self.valid_bg_list = self.valid_bg_list[: fg_have_bg_num * len(self.valid_fg_list)]
            assert len(self.alpha) <= len(self.valid_bg_list)
            self.valid_bg_list = self.valid_bg_list[: len(self.alpha)]

        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)

    def __len__(self):  
        return len(self.alpha)

class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 alpha_ext=".png",
                 merged_ext=".png",
                 trimap_ext=".png"):
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext

        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase

    def __call__(self, sample):
        image, alpha, trimap, mask = sample['image'][:,:,::-1], sample['alpha'], sample['trimap'], sample['mask']
        
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
     
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        image /= 255.

        if self.phase == "train":
            fg = sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg)

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image']

        if CONFIG.model.trimap_channel == 3:
            sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'][None,...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")
        sample['trimap'][sample['trimap'] < 85] = 0
        sample['trimap'][sample['trimap'] >= 170] = 1
        sample['trimap'][sample['trimap'] >= 85] = 0.5

        sample['mask'] = torch.from_numpy(mask).float()

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int32) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample_ori
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, mask, name = sample['fg'],  sample['alpha'], sample['trimap'], sample['mask'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                mask = cv2.resize(mask, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        mask_crop = mask[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap==128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        sample.update({'fg': fg_crop, 'alpha': alpha_crop, 'trimap': trimap_crop, 'mask': mask_crop, 'bg': bg_crop})
        return sample


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample

        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((0,pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
        sample['mask'] = padded_mask

        return sample


class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

    def __call__(self, sample):
        alpha_ori = sample['alpha']
        h, w = alpha_ori.shape

        max_kernel_size = 30
        alpha = cv2.resize(alpha_ori, (640,640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        ### generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int32).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int32).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int32).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int32).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        ### generate mask
        low = 0.01
        high = 1.0
        thres = random.random() * (high - low) + low
        seg_mask = (alpha >= thres).astype(np.int32).astype(np.uint8)
        random_num = random.randint(0,3)
        if random_num == 0:
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 1:
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 2:
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        elif random_num == 3:
            seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        
        seg_mask = cv2.resize(seg_mask, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['mask'] = seg_mask

        return sample


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class CutMask(object):
    def __init__(self, perturb_prob = 0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            return sample

        mask = sample['mask'] # H x W, trimap 0--255, segmask 0--1, alpha 0--1
        h, w = mask.shape
        perturb_size_h, perturb_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)
        
        mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1+perturb_size_h, y1:y1+perturb_size_w].copy()
        
        sample['mask'] = mask
        return sample


class ScaleFg(object):
    def __init__(self, min_scale_fg_scale=0.5, max_scale_fg_scale=1.0):
        self.min_scale_fg_scale = min_scale_fg_scale
        self.max_scale_fg_scale = max_scale_fg_scale

    def __call__(self, sample):
        scale_factor = np.random.uniform(low=self.min_scale_fg_scale, high=self.max_scale_fg_scale)

        fg, alpha = sample['fg'], sample['alpha']  # np.array(): [H, W, 3] 0 ~ 255 , [H, W] 0.0 ~ 1.0
        h, w = alpha.shape
        scale_h, scale_w = int(h * scale_factor), int(w * scale_factor)

        new_fg, new_alpha = np.zeros_like(fg), np.zeros_like(alpha)
        fg = cv2.resize(fg, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR) 
        alpha = cv2.resize(alpha, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR) 

        if scale_factor <= 1:
            offset_h, offset_w = np.random.randint(h - scale_h + 1), np.random.randint(w - scale_w + 1)
            new_fg[offset_h: offset_h + scale_h, offset_w: offset_w + scale_w, :] = fg
            new_alpha[offset_h: offset_h + scale_h, offset_w: offset_w + scale_w] = alpha
        else:
            offset_h, offset_w = np.random.randint(scale_h - h + 1), np.random.randint(scale_w - w + 1)
            new_fg = fg[offset_h: offset_h + scale_h, offset_w: offset_w + scale_w, :]
            new_alpha = alpha[offset_h: offset_h + scale_h, offset_w: offset_w + scale_w]

        sample['fg'], sample['alpha'] = new_fg, new_alpha
        return sample

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

class DataGenerator(Dataset):
    def __init__(
            self, 
            data, 
            phase="train", 
            crop_size=512, 
            remove_multi_fg=False, 
            min_scale_fg_scale=None, 
            max_scale_fg_scale=None, 
            with_bbox = False, 
            bbox_offset_factor = None,
            return_keys = None,
            random_crop_bbox = None,
            dataset_name = None,
            random_auto_matting = None,
        ):
        self.phase = phase
        # self.crop_size = CONFIG.data.crop_size
        self.crop_size = crop_size
        self.remove_multi_fg = remove_multi_fg
        self.with_bbox = with_bbox
        self.bbox_offset_factor = bbox_offset_factor
        self.alpha = data.alpha
        self.return_keys = return_keys
        self.random_crop_bbox = random_crop_bbox
        self.dataset_name = dataset_name
        self.random_auto_matting = random_auto_matting

        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.merged = []
            self.trimap = []
        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap

        train_trans = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            CutMask(perturb_prob=CONFIG.data.cutmask_prob),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train")
        ]
        if min_scale_fg_scale is not None:
            train_trans.insert(0, ScaleFg(min_scale_fg_scale, max_scale_fg_scale))
        if self.with_bbox:
            train_trans.append(GenBBox(bbox_offset_factor=self.bbox_offset_factor, random_crop_bbox=self.random_crop_bbox, random_auto_matting=self.random_auto_matting))

        test_trans = [ OriginScale(), ToTensor() ]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)

    def select_keys(self, sample):
        new_sample = {}
        for key, val in sample.items():
            if key in self.return_keys:
                new_sample[key] = val
        return new_sample

    def __getitem__(self, idx):
        if self.phase == "train":
            fg = cv2.imread(self.fg[idx % self.fg_num])
            alpha = cv2.imread(self.alpha[idx % self.fg_num], 0).astype(np.float32)/255
            bg = cv2.imread(self.bg[idx], 1)

            if not self.remove_multi_fg:
                fg, alpha, multi_fg = self._composite_fg(fg, alpha, idx)
            else:
                multi_fg = False
            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name, 'multi_fg': multi_fg}

        else:
            image = cv2.imread(self.merged[idx])
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            trimap = cv2.imread(self.trimap[idx], 0)
            mask = (trimap >= 170).astype(np.float32)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'mask': mask, 'image_name': image_name, 'alpha_shape': alpha.shape}

        sample = self.transform(sample)

        if self.return_keys is not None:
            sample = self.select_keys(sample)
        if self.dataset_name is not None:
            sample['dataset_name'] = self.dataset_name
        return sample

    def _composite_fg(self, fg, alpha, idx):
        
        multi_fg = False
        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if  np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)
            multi_fg = True

        if np.random.rand() < 0.25:
            # fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            # alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            fg = cv2.resize(fg, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (1280, 1280), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha, multi_fg

    def __len__(self):
        if self.phase == "train":
            return len(self.bg)
        else:
            return len(self.alpha)


class ResziePad(object):

    def __init__(self, target_size=1024):
        self.target_size = target_size

    def __call__(self, sample):
        _, H, W = sample['image'].shape

        scale = self.target_size * 1.0 / max(H, W)
        new_H, new_W = H * scale, W * scale
        new_W = int(new_W + 0.5)
        new_H = int(new_H + 0.5)

        choice = {'image', 'trimap', 'alpha'} if 'trimap' in sample.keys() else {'image', 'alpha'}
        for key in choice:
            if key in {'image', 'trimap'}:
                sample[key] = F.interpolate(sample[key][None], size=(new_H, new_W), mode='bilinear', align_corners=False)[0]
            else:
                # sample[key] = F.interpolate(sample[key][None], size=(new_H, new_W), mode='nearest')[0]
                sample[key] = F.interpolate(sample[key][None], size=(new_H, new_W), mode='bilinear', align_corners=False)[0]
            padding = torch.zeros([sample[key].shape[0], self.target_size, self.target_size], dtype=sample[key].dtype, device=sample[key].device)
            padding[:, : new_H, : new_W] = sample[key]
            sample[key] = padding

        return sample
    

class Cv2ResziePad(object):

    def __init__(self, target_size=1024):
        self.target_size = target_size

    def __call__(self, sample):
        H, W, _ = sample['image'].shape

        scale = self.target_size * 1.0 / max(H, W)
        new_H, new_W = H * scale, W * scale
        new_W = int(new_W + 0.5)
        new_H = int(new_H + 0.5)

        choice = {'image', 'trimap', 'alpha'} if 'trimap' in sample.keys() and sample['trimap'] is not None else {'image', 'alpha'}
        for key in choice:
            sample[key] = cv2.resize(sample[key], (new_W, new_H), interpolation=cv2.INTER_LINEAR)  # cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC

            if key == 'image':
                padding = np.zeros([self.target_size, self.target_size, sample[key].shape[-1]], dtype=sample[key].dtype)
                padding[: new_H, : new_W, :] = sample[key]
                sample[key] = padding
                sample[key] = sample[key][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) #/ 255.0
            else:
                padding = np.zeros([self.target_size, self.target_size], dtype=sample[key].dtype)
                padding[: new_H, : new_W] = sample[key]
                sample[key] = padding
                sample[key] = sample[key][None].astype(np.float32)
            sample[key] = torch.from_numpy(sample[key])

        return sample
    

class AdobeCompositionTest(Dataset):
    def __init__(self, data_dir, target_size=1024, multi_fg=None):
        self.data_dir = data_dir
        self.file_names = sorted(os.listdir(os.path.join(self.data_dir, 'merged')))
        
        test_trans = [
            ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0)
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(os.path.join(self.data_dir, 'alpha_copy', self.file_names[idx])).convert('L')
        tris = Image.open(os.path.join(self.data_dir, 'trimaps', self.file_names[idx]))
        imgs = Image.open(os.path.join(self.data_dir, 'merged', self.file_names[idx]))
        sample = {
            'ori_h_w': (imgs.size[1], imgs.size[0]),
            'data_type': 'Adobe'
        }

        sample['alpha'] = torchvision.transforms.functional.to_tensor(phas)  # [1, H, W] 0.0 ~ 1.0
        sample['trimap'] = torchvision.transforms.functional.to_tensor(tris) * 255.0
        sample['image'] = torchvision.transforms.functional.to_tensor(imgs)
        sample['image_name'] = 'Adobe_' + self.file_names[idx]

        sample = self.transform(sample)
        sample['trimap'][sample['trimap'] < 85] = 0
        sample['trimap'][sample['trimap'] >= 170] = 1
        sample['trimap'][sample['trimap'] >= 85] = 0.5

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample


class SIMTest(Dataset):
    def __init__(self, data_dir, target_size=1024, multi_fg=None):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(*[data_dir, '*', 'alpha', '*'])))  # [: 10]
        test_trans = [
            ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0)
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(self.file_names[idx]).convert('L')
        # tris = Image.open(self.file_names[idx].replace('alpha', 'trimap'))
        imgs = Image.open(self.file_names[idx].replace('alpha', 'merged'))
        sample = {
            'ori_h_w': (imgs.size[1], imgs.size[0]),
            'data_type': 'SIM'
        }

        sample['alpha'] = torchvision.transforms.functional.to_tensor(phas)  # [1, H, W] 0.0 ~ 1.0
        # sample['trimap'] = torchvision.transforms.functional.to_tensor(tris) * 255.0
        sample['image'] = torchvision.transforms.functional.to_tensor(imgs)
        sample['image_name'] = 'SIM_{}_{}'.format(self.file_names[idx].split('/')[-3], self.file_names[idx].split('/')[-1])

        sample = self.transform(sample)
        # sample['trimap'][sample['trimap'] < 85] = 0
        # sample['trimap'][sample['trimap'] >= 170] = 1
        # sample['trimap'][sample['trimap'] >= 85] = 0.5

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample
    

class RW100Test(Dataset):
    def __init__(self, data_dir, target_size=1024, multi_fg=None):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(*[data_dir, 'mask', '*'])))

        self.name_to_idx = dict()
        for idx, file_name in enumerate(self.file_names):
            self.name_to_idx[file_name.split('/')[-1].split('.')[0]] = idx
            
        test_trans = [
            ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0, train_or_test='test', dataset_type='RW100')
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(self.file_names[idx]).convert('L')
        imgs = Image.open(self.file_names[idx].replace('mask', 'image')[:-6] + '.jpg')
        sample = {
            'ori_h_w': (imgs.size[1], imgs.size[0]),
            'data_type': 'RW100'
        }

        sample['alpha'] = torchvision.transforms.functional.to_tensor(phas)  # [1, H, W] 0.0 ~ 1.0
        sample['image'] = torchvision.transforms.functional.to_tensor(imgs)
        sample['image_name'] = 'RW100_' + self.file_names[idx].split('/')[-1]
        
        sample = self.transform(sample)

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample
    
    
class AIM500Test(Dataset):
    def __init__(self, data_dir, target_size=1024, multi_fg=None):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(*[data_dir, 'original', '*'])))

        self.name_to_idx = dict()
        for idx, file_name in enumerate(self.file_names):
            self.name_to_idx[file_name.split('/')[-1].split('.')[0]] = idx

        test_trans = [
            ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0)
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(self.file_names[idx].replace('original', 'mask').replace('jpg', 'png')).convert('L')
        # tris = Image.open(self.file_names[idx].replace('original', 'trimap').replace('jpg', 'png')).convert('L')
        imgs = Image.open(self.file_names[idx])
        sample = {
            'ori_h_w': (imgs.size[1], imgs.size[0]),
            'data_type': 'AIM500'
        }

        sample['alpha'] = torchvision.transforms.functional.to_tensor(phas)  # [1, H, W] 0.0 ~ 1.0
        # sample['trimap'] = torchvision.transforms.functional.to_tensor(tris) * 255.0
        sample['image'] = torchvision.transforms.functional.to_tensor(imgs)
        sample['image_name'] = 'AIM500_' + self.file_names[idx].split('/')[-1]

        sample = self.transform(sample)
        # sample['trimap'][sample['trimap'] < 85] = 0
        # sample['trimap'][sample['trimap'] >= 170] = 1
        # sample['trimap'][sample['trimap'] >= 85] = 0.5

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample


class RWP636Test(Dataset):
    def __init__(self, data_dir, target_size=1024, multi_fg=None):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(*[data_dir, 'image', '*'])))

        self.name_to_idx = dict()
        for idx, file_name in enumerate(self.file_names):
            self.name_to_idx[file_name.split('/')[-1].split('.')[0]] = idx

        test_trans = [
            ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0)
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(self.file_names[idx].replace('image', 'alpha').replace('jpg', 'png')).convert('L')
        imgs = Image.open(self.file_names[idx])
        sample = {
            'ori_h_w': (imgs.size[1], imgs.size[0]),
            'data_type': 'RWP636'
        }

        sample['alpha'] = torchvision.transforms.functional.to_tensor(phas)  # [1, H, W] 0.0 ~ 1.0
        sample['image'] = torchvision.transforms.functional.to_tensor(imgs)
        sample['image_name'] = 'RWP636_' + self.file_names[idx].split('/')[-1]

        sample = self.transform(sample)

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample


class AM2KTest(Dataset):
    def __init__(self, data_dir, target_size=1024, multi_fg=None):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(*[data_dir, 'validation/original', '*'])))
        test_trans = [
            ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0)
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(self.file_names[idx].replace('original', 'mask').replace('jpg', 'png')).convert('L')
        # tris = Image.open(self.file_names[idx].replace('original', 'trimap').replace('jpg', 'png')).convert('L')
        imgs = Image.open(self.file_names[idx])
        sample = {
            'ori_h_w': (imgs.size[1], imgs.size[0]),
            'data_type': 'AM2K'
        }

        sample['alpha'] = torchvision.transforms.functional.to_tensor(phas)  # [1, H, W] 0.0 ~ 1.0
        # sample['trimap'] = torchvision.transforms.functional.to_tensor(tris) * 255.0
        sample['image'] = torchvision.transforms.functional.to_tensor(imgs)
        sample['image_name'] = 'AM2K_' + self.file_names[idx].split('/')[-1]

        sample = self.transform(sample)
        # sample['trimap'][sample['trimap'] < 85] = 0
        # sample['trimap'][sample['trimap'] >= 170] = 1
        # sample['trimap'][sample['trimap'] >= 85] = 0.5

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample


class P3M500Test(Dataset):
    def __init__(self, data_dir, target_size=1024, multi_fg=None):
        self.data_dir = data_dir
        self.file_names = sorted(glob.glob(os.path.join(*[data_dir, 'original_image', '*'])))

        self.name_to_idx = dict()
        for idx, file_name in enumerate(self.file_names):
            self.name_to_idx[file_name.split('/')[-1].split('.')[0]] = idx

        test_trans = [
            ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0)
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.file_names)

    def __getitem__(self, idx):
        phas = Image.open(self.file_names[idx].replace('original_image', 'mask').replace('jpg', 'png')).convert('L')
        # tris = Image.open(self.file_names[idx].replace('original_image', 'trimap').replace('jpg', 'png')).convert('L')
        imgs = Image.open(self.file_names[idx])
        sample = {
            'ori_h_w': (imgs.size[1], imgs.size[0]),
            'data_type': 'P3M500'
        }

        sample['alpha'] = torchvision.transforms.functional.to_tensor(phas)  # [1, H, W] 0.0 ~ 1.0
        # sample['trimap'] = torchvision.transforms.functional.to_tensor(tris) * 255.0
        sample['image'] = torchvision.transforms.functional.to_tensor(imgs)
        sample['image_name'] = 'P3M500_' + self.file_names[idx].split('/')[-1]

        sample = self.transform(sample)
        # sample['trimap'][sample['trimap'] < 85] = 0
        # sample['trimap'][sample['trimap'] >= 170] = 1
        # sample['trimap'][sample['trimap'] >= 85] = 0.5

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample


class MattingTest(Dataset):
    def __init__(
        self, 
        data_type,
        data_dir,
        image_sub_path,
        alpha_sub_path,
        trimpa_sub_path=None,
        target_size=1024, 
        multi_fg=None,
    ):
        self.data_type = data_type
        self.data_dir = data_dir

        self.image_paths = sorted(glob.glob(os.path.join(*[data_dir, image_sub_path])))
        self.alpha_paths = sorted(glob.glob(os.path.join(*[data_dir, alpha_sub_path])))
        self.trimpa_paths = sorted(glob.glob(os.path.join(*[data_dir, trimpa_sub_path]))) if trimpa_sub_path is not None else None

        self.name_to_idx = dict()
        for idx, file_name in enumerate(self.image_paths):
            self.name_to_idx[file_name.split('/')[-1].split('.')[0]] = idx

        test_trans = [
            Cv2ResziePad(target_size=target_size),
            GenBBox(bbox_offset_factor=0)
        ]
        self.transform = transforms.Compose(test_trans)
        self.multi_fg = multi_fg

    def __len__(self):  # 1000
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])
        sample = {
            'image': img.astype(np.float32) / 255,
            'alpha': cv2.imread(self.alpha_paths[idx], 0).astype(np.float32) / 255,
            'trimap': cv2.imread(self.trimpa_paths[idx], 0) if self.trimpa_paths is not None else None,
            'ori_h_w': (img.shape[0], img.shape[1]),
            'data_type': self.data_type,
            'image_name': self.data_type + '_' + self.image_paths[idx].split('/')[-1]
        }

        sample = self.transform(sample)
        if self.trimpa_paths is not None:
            sample['trimap'][sample['trimap'] < 85] = 0
            sample['trimap'][sample['trimap'] >= 170] = 1
            sample['trimap'][sample['trimap'] >= 85] = 0.5
        else:
            del sample['trimap']

        if self.multi_fg is not None:
            sample['multi_fg'] = torch.tensor(self.multi_fg)

        return sample


def adobe_composition_collate_fn(batch):
    new_batch = defaultdict(list)
    for sub_batch in batch:
        for key in sub_batch.keys():
            new_batch[key].append(sub_batch[key])
    for key in new_batch: 
        if isinstance(new_batch[key][0], torch.Tensor):
            new_batch[key] = torch.stack(new_batch[key])
    return dict(new_batch)


def build_d2_test_dataloader(
    dataset,
    mapper=None,
    total_batch_size=None,
    local_batch_size=None,
    num_workers=0,
    collate_fn=None
):

    assert (total_batch_size is None) != (
        local_batch_size is None
    ), "Either total_batch_size or local_batch_size must be specified"

    world_size = comm.get_world_size()

    if total_batch_size is not None:
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size

    if local_batch_size is not None:
        batch_size = local_batch_size

    logger = logging.getLogger(__name__)
    if batch_size != 1:
        logger.warning(
            "When testing, batch size is set to 1. "
            "This is the only mode that is supported for d2."
        )

    return build_detection_test_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=None,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


class AdobeCompositionEvaluator(DatasetEvaluator):

    def __init__(
        self, 
        save_eval_results_step=-1, 
        output_dir=None, 
        eval_dataset_type=['Adobe'],
        distributed=True,
        eval_w_sam_hq_mask = False,
    ):  

        self.save_eval_results_step = save_eval_results_step
        self.output_dir = output_dir
        self.eval_index = 0
        self.eval_dataset_type = eval_dataset_type
        self.eval_w_sam_hq_mask = eval_w_sam_hq_mask

        self._distributed = distributed
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self.eval_metric = dict()
        for i in self.eval_dataset_type:
            self.eval_metric[i + '_MSE'] = []
            self.eval_metric[i + '_SAD'] = []
            self.eval_metric[i + '_MAD'] = []
            self.eval_metric[i + '_Grad'] = []
            self.eval_metric[i + '_Conn'] = []

        os.makedirs(self.output_dir, exist_ok=True) if self.output_dir is not None else None

    def process(self, inputs, outputs):
        """
        Args:
            inputs: {'alpha', 'trimap', 'image', 'bbox', 'image_name'}
            outputs: [1, 1, H, W] 0. ~ 1.
        """

        # crop the black pad area
        assert inputs['image'].shape[-1] == inputs['image'].shape[-2] == 1024 and len(inputs['ori_h_w']) == 1
        inputs['ori_h_w'] = inputs['ori_h_w'][0]
        before_pad_h, before_pad_w = int(1024 / max(inputs['ori_h_w']) * inputs['ori_h_w'][0] + 0.5), int(1024 / max(inputs['ori_h_w']) * inputs['ori_h_w'][1] + 0.5)
        inputs['image'] = inputs['image'][:, :, :before_pad_h, :before_pad_w]
        inputs['alpha'] = inputs['alpha'][:, :, :before_pad_h, :before_pad_w]

        if self.eval_w_sam_hq_mask:
            outputs, samhq_low_res_masks = outputs[0][:, :, :before_pad_h, :before_pad_w], outputs[1][:, :, :before_pad_h, :before_pad_w]
            pred_alpha, label_alpha, samhq_low_res_masks = outputs.cpu().numpy(), inputs['alpha'].numpy(), (samhq_low_res_masks > 0).float().cpu()
        else:
            outputs = outputs[:, :, :before_pad_h, :before_pad_w]
            pred_alpha, label_alpha = outputs.cpu().numpy(), inputs['alpha'].numpy()

        # if 'trimap' in inputs.keys():
        #     inputs['trimap'] = inputs['trimap'][:, :, :before_pad_h, :before_pad_w]
        #     trimap = inputs['trimap'].numpy()
        #     assert np.max(trimap) <= 1 and len(np.unique(trimap)) <= 3
        #     sad_loss_unknown = compute_sad_loss(pred_alpha, label_alpha, trimap, area='unknown')
        #     mse_loss_unknown = compute_mse_loss(pred_alpha, label_alpha, trimap, area='unknown')

        #     self.eval_metric[inputs['data_type'][0] + '_unknown_mse (1e-3)'].append(mse_loss_unknown)
        #     self.eval_metric[inputs['data_type'][0] + '_unknown_sad (1e3)'].append(sad_loss_unknown)

        # calculate loss
        assert np.max(pred_alpha) <= 1 and np.max(label_alpha) <= 1
        eval_pred = np.uint8(pred_alpha[0, 0] * 255.0 + 0.5) * 1.0
        eval_gt = label_alpha[0, 0] * 255.0

        detailmap = np.zeros_like(eval_gt) + 128
        mse_loss_ = compute_mse_loss(eval_pred, eval_gt, detailmap)
        sad_loss_ = compute_sad_loss(eval_pred, eval_gt, detailmap)[0]
        mad_loss_ = compute_mad_loss(eval_pred, eval_gt, detailmap)
        grad_loss_ = compute_gradient_loss(eval_pred, eval_gt, detailmap)
        conn_loss_ = compute_connectivity_error(eval_pred, eval_gt, detailmap)

        self.eval_metric[inputs['data_type'][0] + '_MSE'].append(mse_loss_)
        self.eval_metric[inputs['data_type'][0] + '_SAD'].append(sad_loss_)
        self.eval_metric[inputs['data_type'][0] + '_MAD'].append(mad_loss_)
        self.eval_metric[inputs['data_type'][0] + '_Grad'].append(grad_loss_)
        self.eval_metric[inputs['data_type'][0] + '_Conn'].append(conn_loss_)

        # vis results
        if self.save_eval_results_step != -1 and self.eval_index % self.save_eval_results_step == 0:
            if self.eval_w_sam_hq_mask:
                self.save_vis_results(inputs, pred_alpha, samhq_low_res_masks)
            else:
                self.save_vis_results(inputs, pred_alpha)
        self.eval_index += 1

    def save_vis_results(self, inputs, pred_alpha, samhq_low_res_masks=None):

        # image
        image = inputs['image'][0].permute(1, 2, 0) * 255.0
        l, u, r, d = int(inputs['bbox'][0, 0, 0].item()), int(inputs['bbox'][0, 0, 1].item()), int(inputs['bbox'][0, 0, 2].item()), int(inputs['bbox'][0, 0, 3].item())
        red_line = torch.tensor([[255., 0., 0.]], device=image.device, dtype=image.dtype)
        image[u: d, l, :] = red_line
        image[u: d, r, :] = red_line
        image[u, l: r, :] = red_line
        image[d, l: r, :] = red_line
        image = np.uint8(image.numpy())

        # trimap, pred_alpha, label_alpha
        save_results = [image]

        choice = [inputs['trimap'], torch.from_numpy(pred_alpha), inputs['alpha']] if 'trimap' in inputs.keys() else [torch.from_numpy(pred_alpha), inputs['alpha']]
        for val in choice:
            val = val[0].permute(1, 2, 0).repeat(1, 1, 3) * 255.0 + 0.5  # +0.5 and int() = round()
            val = np.uint8(val.numpy())
            save_results.append(val)

        if samhq_low_res_masks is not None:
            save_results.append(np.uint8(samhq_low_res_masks[0].permute(1, 2, 0).repeat(1, 1, 3).numpy() * 255.0))

        save_results = np.concatenate(save_results, axis=1)
        save_name = os.path.join(self.output_dir, inputs['image_name'][0])
        Image.fromarray(save_results).save(save_name.replace('.jpg', '.png'))

    def evaluate(self):
        
        if self._distributed:
            comm.synchronize()
            eval_metric = comm.gather(self.eval_metric, dst=0)

            if not comm.is_main_process():
                return {}
            
            merges_eval_metric = defaultdict(list)
            for sub_eval_metric in eval_metric:
                for key, val in sub_eval_metric.items():
                    merges_eval_metric[key] += val
            eval_metric = merges_eval_metric

        else:
            eval_metric = self.eval_metric

        eval_results = {}

        for key, val in eval_metric.items():
            if len(val) != 0:
                # if 'mse' in key:
                #     eval_results[key] = np.array(val).mean() * 1e3
                # else:
                #     assert 'sad' in key
                #     eval_results[key] = np.array(val).mean() / 1e3
                eval_results[key] = np.array(val).mean()

        return eval_results


if __name__ == '__main__':
    pass