from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random

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

    def get_all_files(self, datasets_root='/mnt/data/ziyiliu/virtual_staining/he2pas_uniform/'):
        pkl_path = os.path.join(datasets_root, '{}.pkl'.format(self.mode))
        if os.path.exists(pkl_path):
            print('{} existed!'.format(pkl_path))
            with open(pkl_path, 'rb') as f:
                image_paths = pickle.load(f)
            return image_paths

        paired_images = []
        data_root = os.path.join(datasets_root, self.mode)
        handles = os.listdir(data_root)
        for handle in handles:
            handle = os.path.join(data_root, handle)
            dirs = os.listdir(handle)
            for data_dir in dirs:
                if '.' in data_dir:
                    continue
                data_dir = os.path.join(handle, data_dir)
                paired_tuple = ['', '']
                for file_name in os.listdir(data_dir):
                    if file_name[:2] == 'he':
                        paired_tuple[0] = os.path.join(data_dir, file_name)
                    elif file_name[:2] == 'pa':
                        paired_tuple[1] = os.path.join(data_dir, file_name)
                paired_images.append(paired_tuple)

        if self.mode == 'train' or self.mode == 'valid':
            random.shuffle(paired_images)
            train_image_paths = paired_images[:int(len(paired_images) * 0.9)]
            valid_image_paths = paired_images[int(len(paired_images) * 0.9):]
            train_pkl_path = os.path.join(datasets_root, 'train.pkl')
            valid_pkl_path = os.path.join(datasets_root, 'val.pkl')
            with open(train_pkl_path, 'wb') as f:
                pickle.dump(train_image_paths, f)
            with open(valid_pkl_path, 'wb') as f:
                pickle.dump(valid_image_paths, f)
            print('process completed!!')
        elif self.mode == 'test':
            test_pkl_path = os.path.join(datasets_root, 'test.pkl')
            with open(test_pkl_path, 'wb') as f:
                pickle.dump(paired_images, f)
            print('process completed!!')
        else:
            print('wrong mode!!')
            exit()


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
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        image_file = self.image_files[index]['image_path']

        # HE
        heorpas = 0
        he_name = image_file[heorpas].split('/')[-1]
        he_f = os.path.join('/mnt/data/ziyiliu/virtual_staining/he2pas_uniform/restored_test', he_name)
        # he_f = os.path.join(self.root, image_file[heorpas])
        he_img = self.load_gt_image(he_f)
        he_img = (he_img[..., ::-1] / 255.0).astype(np.float32)

        # PAS
        heorpas = 1
        # pas_name = image_file[heorpas].split('/')[-1]
        pas_f = os.path.join(self.root, image_file[heorpas])
        pas_img = self.load_gt_image(pas_f)
        pas_img = (pas_img[..., ::-1] / 255.0).astype(np.float32)

        h, w, _ = he_img.shape

        prompt = "Translate the Hematoxylin and Eosin (H&E) image to Periodic Acid-Schiff (PAS) stained image. "

        # BGR to RGB, [0, 1]
        lq = he_img[..., ::-1].astype(np.float32)

        # BGR to RGB, [-1, 1]
        gt = (pas_img[..., ::-1] * 2 - 1).astype(np.float32)

        return gt, lq, prompt, he_name

    def __len__(self) -> int:
        return len(self.image_files)
