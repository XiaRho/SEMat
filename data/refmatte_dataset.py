import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import random
import imgaug.augmenters as iaa
import numbers
import math


def random_interp():
    return np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])

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

        fg = cv2.warpAffine(fg, M, (cols, rows), flags=random_interp() + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows), flags=random_interp() + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample

    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

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
    

class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    def __call__(self, sample):
        alpha = sample['alpha']
        h, w = alpha.shape

        max_kernel_size = max(30, int((min(h,w) / 2048) * 30))

        ### generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int32).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int32).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        return sample
    

class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(1024, 1024)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2

    def __call__(self, sample):
        fg, alpha, trimap, name = sample['fg'],  sample['alpha'], sample['trimap'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=random_interp())
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=random_interp())
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=random_interp())
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=random_interp())
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

        if len(np.where(trimap==128)[0]) == 0:
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=random_interp())
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=random_interp())
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=random_interp())
        
        sample.update({'fg': fg_crop, 'alpha': alpha_crop, 'trimap': trimap_crop, 'bg': bg_crop})
        return sample
    

class Composite_Seg(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        image = fg
        sample['image'] = image
        return sample
    

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test", real_world_aug = False):
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.mean = torch.tensor([0.0, 0.0, 0.0]).view(3,1,1)
        self.std = torch.tensor([1.0, 1.0, 1.0]).view(3,1,1)
        self.phase = phase
        if real_world_aug:
            self.RWA = iaa.SomeOf((1, None), [
                iaa.LinearContrast((0.6, 1.4)),
                iaa.JpegCompression(compression=(0, 60)),
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
            ], random_order=True)
        else:
            self.RWA = None
    
    def get_box_from_alpha(self, alpha_final):
        bi_mask = np.zeros_like(alpha_final)
        bi_mask[alpha_final>0.5] = 1
        #bi_mask[alpha_final<=0.5] = 0
        fg_set = np.where(bi_mask != 0)
        if len(fg_set[1]) == 0 or len(fg_set[0]) == 0:
            x_min = random.randint(1, 511)
            x_max = random.randint(1, 511) + x_min
            y_min = random.randint(1, 511)
            y_max = random.randint(1, 511) + y_min
        else:
            x_min = np.min(fg_set[1])
            x_max = np.max(fg_set[1])
            y_min = np.min(fg_set[0])
            y_max = np.max(fg_set[0])
        bbox = np.array([x_min, y_min, x_max, y_max])
        #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
        #cv2.imwrite('../outputs/test.jpg', image)
        #cv2.imwrite('../outputs/test_gt.jpg', alpha_single)
        return bbox

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha, trimap = sample['image'][:,:,::-1], sample['alpha'], sample['trimap']
        
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        
        bbox = self.get_box_from_alpha(alpha)

        if self.phase == 'train' and self.RWA is not None and np.random.rand() < 0.5:
            image[image > 255] = 255
            image[image < 0] = 0
            image = np.round(image).astype(np.uint8)
            image = np.expand_dims(image, axis=0)
            image = self.RWA(images=image)
            image = image[0, ...]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1
        #image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 3)
        #cv2.imwrite(os.path.join('outputs', 'img_bbox.png'), image.astype('uint8'))
        # normalize image
        image /= 255.

        if self.phase == "train":
            # convert GBR images to RGB
            fg = sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
            del sample['image_name']
        
        sample['boxes'] = torch.from_numpy(bbox).to(torch.float)[None,...]

        sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['trimap'] = sample['trimap'][None,...].float()

        return sample


class RefMatteData(Dataset):
    def __init__(
        self, 
        data_root_path,
        num_ratio = 0.34,
    ):
        self.data_root_path = data_root_path
        self.num_ratio = num_ratio

        self.rim_img = [os.path.join(data_root_path, name) for name in sorted(os.listdir(data_root_path))]
        self.rim_pha = [os.path.join(data_root_path.replace('img', 'mask'), name) for name in sorted(os.listdir(data_root_path.replace('img', 'mask')))]
        self.rim_num = len(self.rim_pha)

        self.transform_spd = transforms.Compose([
            RandomAffine(degrees=30, scale=[0.8, 1.5], shear=10, flip=0.5),
            GenTrimap(),
            RandomCrop((1024, 1024)),
            Composite_Seg(),
            ToTensor(phase="train", real_world_aug=False)
        ])

    def __getitem__(self, idx):
        if self.num_ratio is not None:
            if self.num_ratio < 1.0 or idx >= self.rim_num:
                idx = np.random.randint(0, self.rim_num)
        alpha = cv2.imread(self.rim_pha[idx % self.rim_num], 0).astype(np.float32)/255
        alpha_img_name = self.rim_pha[idx % self.rim_num].split('/')[-1]
        fg_img_name = alpha_img_name[:-6] + '.jpg'

        fg = cv2.imread(os.path.join(self.data_root_path, fg_img_name))

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (1280, 1280), interpolation=random_interp())
            alpha = cv2.resize(alpha, (1280, 1280), interpolation=random_interp())

        image_name = alpha_img_name  # os.path.split(self.rim_img[idx % self.rim_num])[-1]
        sample = {'fg': fg, 'alpha': alpha, 'bg': fg, 'image_name': image_name}
        sample = self.transform_spd(sample)

        converted_sample = {
            'image': sample['image'],
            'trimap': sample['trimap'] / 2.0,
            'alpha': sample['alpha'],
            'bbox': sample['boxes'],
            'dataset_name': 'RefMatte',
            'multi_fg': False,
        }
        return converted_sample

    def __len__(self):
        if self.num_ratio is not None:
            return int(self.rim_num * self.num_ratio)  # 112506 * 0.34 = 38252 (COCONut_num-38251 + 1)
        else:
            return self.rim_num  # 112506


    
if __name__ == '__main__':
    dataset = RefMatteData(
        data_root_path = '/data/my_path_b/public_data/data/matting/RefMatte/RefMatte/train/img', 
        num_ratio=0.34,
    )
    data = dataset[0]
    '''
    fg torch.Size([3, 1024, 1024]) tensor(-2.1179) tensor(2.6400)
    alpha torch.Size([1, 1024, 1024]) tensor(0.) tensor(1.)
    bg torch.Size([3, 1024, 1024]) tensor(-2.1179) tensor(2.6400)
    trimap torch.Size([1, 1024, 1024]) 0.0 or 1.0 or 2.0
    image torch.Size([3, 1024, 1024]) tensor(-2.1179) tensor(2.6400)
    boxes torch.Size([1, 4]) tensor(72.) tensor(676.)  0.0~1024.0

    COCONut:
        image torch.Size([3, 1024, 1024]) tensor(0.0006) tensor(0.9991)
        trimap torch.Size([1, 1024, 1024]) 0.0 or 0.5 or 1.0
        alpha torch.Size([1, 1024, 1024]) tensor(0.) tensor(1.)
        bbox torch.Size([1, 4]) tensor(0.) tensor(590.)
        dataset_name: 'COCONut'
    '''
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            print(key, val.shape, torch.min(val), torch.max(val))
        else:
            print(key, val.shape)