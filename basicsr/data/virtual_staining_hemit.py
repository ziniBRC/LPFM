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
    
    def get_all_files(self, datasets_root='/mnt/data/ziyiliu/virtual_staining/HEMIT/'):
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
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        image_file = self.image_files[index]
        input_path = os.path.join(self.root, image_file["image_path"][0])
        input_name = input_path.split('/')[-1]
        input_image = cv2.imread(input_path)
        input_image = Image.fromarray(input_image)

        # target
        target_path = os.path.join(self.root, self.image_paths[index][1])
        target_name = target_path.split('/')[-1]
        target_image = cv2.imread(target_path)
        target_image = Image.fromarray(target_image)

        he_img = input_image
        mit_img = target_image
        h, w, _ = he_img.shape

        # some paired images are used for restoration, some paired images are used for virtual staining 
        task = 'restoration'
        random_selector = np.random.uniform()
        if random_selector < 0:
            prompt = "Restore the low-quality hematoxylin and eosin stained image. "
            task = 'restoration'
            img_gt = he_img
        elif random_selector < 0:
            prompt = "Restore the low-quality multiplex-immunohistochemistry image. "
            task = 'restoration'
            img_gt = mit_img
        else:
            prompt = "Translate the hematoxylin & eosin image to multiplex-immunohistochemistry image. "
            task = 'virtual staining'

        if task == 'restoration':
            # ------------------------ generate lq image ------------------------ #
            # blur
            kernel, kernel_type, density = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None,
            )
            img_lq = cv2.filter2D(img_gt, -1, kernel)
            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(
                img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
            )
            # jpeg compression
            if self.jpeg_range is not None:
                img_lq, random_quality = random_add_jpg_compression(img_lq, self.jpeg_range)
            # noise
            if self.noise_range is not None:
                img_lq, random_sigma = random_add_gaussian_noise(img_lq, self.noise_range)

            # resize to original size
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

            if np.random.uniform() < 0.5 and self.stage == 2:
                prompt += f"degradation type and density: blur {kernel_type} {round(density[0], 2)} {round(density[1], 2)}, downsample {round(scale, 2)}, noise sigma {round(random_sigma, 2)}, jpeg_range {random_quality} "

            # BGR to RGB, [-1, 1]
            gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
            # BGR to RGB, [0, 1]
            lq = img_lq[..., ::-1].astype(np.float32)
        else:
            # af to he
            # BGR to RGB, [-1, 1]he_img
            gt = (mit_img[..., ::-1] * 2 - 1).astype(np.float32)
            # BGR to RGB, [0, 1]
            lq = he_img[..., ::-1].astype(np.float32)
        return gt, lq, prompt

    def __len__(self) -> int:
        return len(self.image_files)
