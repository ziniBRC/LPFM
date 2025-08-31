from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
from torchvision import transforms
import numpy as np
import os 
import pickle
import cv2
from PIL import Image
import torch.utils.data as data

from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        mode: str,
        stage: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.root = file_list
        self.file_list = file_list
        self.mode = mode
        self.stage = stage
        image_paths = self.get_all_files(self.root)
        self.image_files = self.load_file_list(image_paths)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
    
    def load_file_list(self, image_paths):
        files = []
        for path in image_paths:
            if path:
                # image_path = self.root + '/images' + path
                files.append({"image_path": path, "prompt": ""})
        return files
    
    def get_all_files(self, datasets_root='/mnt/data/ziyiliu/virtual_staining/HEMIT/'):
        if self.mode == 'all':
            print('mode is all, read all train, val, test data!!')
            all_path = []
            pkl_path = os.path.join(datasets_root, '{}.pkl'.format('train'))
            with open(pkl_path, 'rb') as f:
                image_paths = pickle.load(f)
            all_path += image_paths
            pkl_path = os.path.join(datasets_root, '{}.pkl'.format('val'))
            with open(pkl_path, 'rb') as f:
                image_paths = pickle.load(f)
            all_path += image_paths
            pkl_path = os.path.join(datasets_root, '{}.pkl'.format('test'))
            with open(pkl_path, 'rb') as f:
                image_paths = pickle.load(f)
            all_path += image_paths
            return all_path

        pkl_path = os.path.join(datasets_root, '{}.pkl'.format(self.mode))
        if os.path.exists(pkl_path):
            print('{} existed!'.format(pkl_path))
            with open(pkl_path, 'rb') as f:
                image_paths = pickle.load(f)
            return image_paths

        image_paths = []
        image_names = os.listdir(os.path.join(datasets_root, self.mode, 'input'))
        for image_name in image_names:
            input_image_path = os.path.join(self.mode, 'input', image_name)
            label_image_path = os.path.join(self.mode, 'label', image_name)
            image_paths.append([input_image_path, label_image_path])

        with open(pkl_path, 'wb') as f:
            pickle.dump(image_paths, f)

        return image_paths
    
    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    # print('center crop images')
                    # image = np.array(image)[256:-256, 256:-256, :]
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    resize_func = transforms.RandomResizedCrop(512, (0.3, 0.7))
                    # image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
                    image = resize_func(image)
                    image = np.array(image)
                    
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        if self.mode == 'train':
            index = index // 100

        image_file = self.image_files[index]
        input_path = os.path.join(self.root, image_file["image_path"][0])
        # input_path = input_path.replace('input', 'restored_input')
        input_path = input_path.replace('tif', 'png')

        lq_name = input_path.split('/')[-1]
        input_image = self.load_gt_image(input_path)
        input_image = (input_image[..., ::-1] / 255.0).astype(np.float32)

        # target
        target_path = os.path.join(self.root, image_file["image_path"][1])
        target_name = target_path.split('/')[-1]
        target_image = self.load_gt_image(target_path)
        target_image = (target_image[..., ::-1] / 255.0).astype(np.float32)

        he_img = input_image
        mit_img = target_image
        h, w, _ = he_img.shape

        prompt = "Restore the low-quality multiplex-immunohistochemistry image."

        # BGR to RGB, [-1, 1]
        gt = (mit_img[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        lq = he_img[..., ::-1].astype(np.float32)

        return gt, lq, prompt, lq_name

    def __len__(self) -> int:
        if self.mode == 'train':
            return len(self.image_files) * 100
        else:
            return len(self.image_files) 