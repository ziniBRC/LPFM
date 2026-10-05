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

    def get_all_files(self, datasets_root='/mnt/data/ziyiliu/virtual_staining/af2he/'):
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
                    if file_name[:2] == '1_':
                        paired_tuple[0] = os.path.join(data_dir, file_name)
                    elif file_name[:2] == '2_':
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
        aforhe = 0 # AF
        image_file = self.image_files[index]
        image_paths = image_file["image_path"]
        af_name = image_paths[aforhe].split('/')[-1]
        af_f = image_paths[aforhe].replace('\n', '').replace('/media/user/data/af2he/', self.root)
        af_img = self.load_gt_image(af_f)
        af_img = (af_img[..., ::-1] / 255.0).astype(np.float32)

        aforhe = 1 # HE
        he_name = image_paths[aforhe].split('/')[-1]
        he_f = image_paths[aforhe].replace('\n', '').replace('/media/user/data/af2he/', self.root)
        he_img = self.load_gt_image(he_f)
        he_img = (he_img[..., ::-1] / 255.0).astype(np.float32)

        h, w, _ = af_img.shape

        # some paired images are used for restoration, some paired images are used for virtual staining 
        task = 'restoration'
        random_selector = np.random.uniform()
        if random_selector < 0:
            prompt = "Restore the low-quality autofluorescence image. "
            task = 'restoration'
            img_gt = af_img
        elif random_selector < 0:
            prompt = "Restore the low-quality hematoxylin and eosin stained image. "
            task = 'restoration'
            img_gt = he_img
        else:
            prompt = "Translate the autofluorescence image to hematoxylin and eosin stained image. "
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
            gt = (he_img[..., ::-1] * 2 - 1).astype(np.float32)
            # BGR to RGB, [0, 1]
            lq = af_img[..., ::-1].astype(np.float32)
            lq = 1 - lq
        return gt, lq, prompt

    def __len__(self) -> int:
        return len(self.image_files)
