from typing import List
from argparse import ArgumentParser
import random
import os 
import numpy as np
import torch
import gradio as gr
from argparse import ArgumentParser
import matplotlib.pyplot as plt 
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
import lpips
import tifffile

from modules.model import SwinIR
from modules.utils.common import instantiate_from_config, calculate_psnr_pt, to
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate.utils import set_seed

from modules.model.cldm import ControlLDM
from modules.model.swinir import SwinIR
from modules.inference.pretrained_models import MODELS
from modules.utils.common import instantiate_from_config, load_model_from_url
from modules.model.gaussian_diffusion import Diffusion
from modules.pipeline import SwinIRPipeline
from modules.utils.caption import (
    EmptyCaptioner,
    LLaVACaptioner,
    RAMCaptioner,
    LLAVA_AVAILABLE,
    RAM_AVAILABLE,
)

torch.set_grad_enabled(False)

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    if cfg.train.resume:
        swinir.load_state_dict(
            torch.load(cfg.train.resume, map_location="cpu"), strict=True
        )
        if accelerator.is_local_main_process:
            print(f"strictly load weight from checkpoint: {cfg.train.resume}")
    else:
        if accelerator.is_local_main_process:
            print("initialize from scratch")

    # Setup optimizer:
    opt = torch.optim.AdamW(
        swinir.parameters(), lr=cfg.train.learning_rate, weight_decay=0
    )

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.val)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        drop_last=False,
    )

    lq_save_dir = '/mnt/data/ziyiliu/virtual_staining/he2pas_uniform/degraded_test'
    lq_gt_dir = '/mnt/data/ziyiliu/virtual_staining/he2pas_uniform/test_gt'
    i = 0
    if os.path.exists(lq_save_dir) is False:
        os.makedirs(lq_save_dir)

    for batch in loader:
        gt, lq, prompt, he_f, pas_f = batch
        lq = lq.detach().cpu().numpy()
        gt = (gt.detach().cpu().numpy() + 1) / 2 
        # image_name = name[0].split('.')[0]
        he_f = he_f[0].replace('test', 'degraded_test')
        os.makedirs(os.path.dirname(he_f), exist_ok=True)
        plt.imsave(he_f, lq[0, :])

        pas_f = pas_f[0].replace('test', 'degraded_test')
        os.makedirs(os.path.dirname(pas_f), exist_ok=True)
        plt.imsave(pas_f, gt[0, :])

        # plt.imsave(os.path.join(lq_save_dir, f'{image_name}.png'), lq[0, :])
        # tifffile.imwrite(os.path.join(lq_save_dir, name[0]), lq[0, :])
        print(f'save {he_f}')
        i += 1
        # if i == 200:
        #     break
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='configs/train/test_stage2_hemit_degraded.yaml',type=str, required=True)

    args = parser.parse_args()

    main(args)
