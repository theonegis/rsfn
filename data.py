from pathlib import Path
import numpy as np
import rasterio
import math
from enum import Enum, auto, unique

import torch
from torch.utils.data import Dataset

from utils import make_tuple

NUM_BANDS = 6
FINE_PREFIX = "LC08"
COARSE_PREFIX = "MOD09GA"
SCALE_FACTOR = 16
NODATA_VALUE = -9999
PATCH_SIZE = 256


@unique
class Mode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    PREDICTION = auto()


def get_pair_path(directory: Path, mode: Mode):
    # 将一组数据集按照规定的顺序组织好
    # For Tianjin and Nebraska Datasets
    dt_prev, dt_pred, dt_next = directory.stem.split('-')
    orders = [
        f'{FINE_PREFIX}*{dt_prev}*[!QA].tif',
        f'{FINE_PREFIX}*{dt_next}*[!QA].tif',
        f'{COARSE_PREFIX}.{dt_pred}*.tif',
        f'{FINE_PREFIX}*{dt_pred}*[!QA].tif'
    ]

    if mode == Mode.PREDICTION:
        del orders[-1]
    paths = []
    for i in orders:
        paths.append(next(directory.glob(i)))

    # For CIA and LGC Datasets
    # prev_label, pred_label, next_label = directory.name.split('-')
    # prev_tokens, pred_tokens, next_tokens = prev_label.split('_'), pred_label.split('_'), next_label.split('_')
    #
    # def match(path: Path):
    #     return {
    #         prev_tokens[0] + prev_tokens[2] in path.stem and 'QA' not in path.stem: 0,
    #         next_tokens[0] + next_tokens[2] in path.stem and 'QA' not in path.stem: 1,
    #         pred_tokens[0] + pred_tokens[1] in path.stem: 2,
    #         pred_tokens[0] + pred_tokens[2] in path.stem and 'QA' not in path.stem: 3
    #     }
    #
    # paths = [None] * 4
    # for f in Path(directory).glob('*.tif'):
    #     try:
    #         paths[match(f)[True]] = f.absolute().resolve()
    #     except KeyError:
    #         continue
    #
    # if mode is Mode.PREDICTION:
    #     del paths[-1]
    return paths


# def random_mask(size):
#     mask = np.ones((1, *size), np.int16)
#     counts = random.randint(10, 30)
#     for i in range(counts):
#         width = random.randint(20, 100)
#         height = random.randint(20, 100)
#         x = random.randint(0, size[0] - 1)
#         y = random.randint(0, size[1] - 1)
#         mask[:, x: x + width, y: y + height] = 0
#     return mask
#
#
# def load_image_pair(directory: Path, mode: Mode):
#     # 按照一定顺序获取给定文件夹下的一组数据
#     paths = get_pair_path(directory, mode)
#     images = []
#     for i in range(len(paths)):
#         if i == 1 or i == 3:
#             continue
#         with rasterio.open(str(paths[i])) as ds:
#             im = ds.read()
#             mask = None
#             if (i == 0 or i == 2) and mode == Mode.TRAINING:
#                 mask = random_mask(im.shape[1:])
#                 im *= mask
#             im[im < 0] = 0
#             images.append(im)
#             if mask is not None:
#                 images.append(mask)
#     return images


def load_image_pair(directory: Path, mode: Mode):
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(directory, mode)
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            im[im < 0] = 0
            images.append(im)
    return images


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, mode=Mode.TRAINING):
        super(PatchSet, self).__init__()
        self.root_dir = image_dir
        self.image_size = make_tuple(image_size)
        self.patch_size = make_tuple(patch_size)
        self.patch_stride = self.patch_size if patch_stride is None else make_tuple(patch_stride)
        self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)
        self.mode = mode

        # 计算出图像进行分块以后的patches的数目
        self.num_patches_x = math.ceil((self.image_size[0] - self.patch_size[0] + 1) / self.patch_stride[0])
        self.num_patches_y = math.ceil((self.image_size[1] - self.patch_size[1] + 1) / self.patch_stride[1])
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

    def map_index(self, index):
        # 将全局的index映射到具体的图像对文件夹索引(id_n)，图像裁剪的列号与行号(id_x, id_y)
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n], self.mode)
        patches = [None] * len(images)
        for i in range(len(patches)):
            im = images[i][:,
                 id_x: (id_x + self.patch_size[0]),
                 id_y: (id_y + self.patch_size[1])]
            patches[i] = (lambda data: torch.from_numpy(data.astype(np.float32)))(im)
            patches[i] = patches[i].mul_(0.0001)
        del images[:]
        del images
        return patches

    def __len__(self):
        return self.num_patches
