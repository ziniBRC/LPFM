
from argparse import ArgumentParser
import numpy as np
import torch
import os 
from argparse import ArgumentParser

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

from diffbir.model import SwinIR
from diffbir.utils.common import instantiate_from_config, calculate_psnr_pt, to
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate.utils import set_seed

from diffbir.model.cldm import ControlLDM
from diffbir.model.swinir import SwinIR
from diffbir.inference.pretrained_models import MODELS
from diffbir.utils.common import instantiate_from_config, load_model_from_url
from diffbir.model.gaussian_diffusion import Diffusion
from diffbir.pipeline import SwinIRPipeline
from diffbir.utils.caption import (
    EmptyCaptioner,
    LLaVACaptioner,
    RAMCaptioner,
    LLAVA_AVAILABLE,
    RAM_AVAILABLE,
)

torch.set_grad_enabled(False)

# This gradio script only support DiffBIR v2.1
parser = ArgumentParser()
parser.add_argument("--captioner", type=str, choices=["none", "ram", "llava"], required=True)
parser.add_argument("--llava_bit", type=str, choices=["4", "8", "16"], default="4")
args = parser.parse_args()

# Set max height and width to constraint inference time for online demo
max_height = 2048
max_width = 2048

tasks = ["sr", "face"]
device = "cuda"
precision = "fp16"
llava_bit = args.llava_bit
# Set captioner to llava or ram to enable auto-caption
captioner = args.captioner

if captioner == "llava":
    assert LLAVA_AVAILABLE
elif captioner == "ram":
    assert RAM_AVAILABLE

# 1. load stage-1 models
swinir: SwinIR = instantiate_from_config(
    OmegaConf.load("configs/inference/swinir.yaml")
)
swinir.load_state_dict(load_model_from_url(MODELS["swinir_realesrgan"]))
swinir.eval().to(device)

face_swinir: SwinIR = instantiate_from_config(
    OmegaConf.load("configs/inference/swinir.yaml")
)
face_swinir.load_state_dict(load_model_from_url(MODELS["swinir_face"]))
face_swinir.eval().to(device)

# 2. load stage-2 model
cldm: ControlLDM = instantiate_from_config(
    OmegaConf.load("configs/inference/cldm.yaml")
)
# 2.1 load pre-trained SD
sd_weight = load_model_from_url(MODELS["sd_v2.1_zsnr"])
unused, missing = cldm.load_pretrained_sd(sd_weight)
print(
    f"load pretrained stable diffusion, "
    f"unused weights: {unused}, missing weights: {missing}"
)
# 2.2 load ControlNet
control_weight = load_model_from_url(MODELS["v2.1"])
cldm.load_controlnet_from_ckpt(control_weight)
print("load controlnet weight")
cldm.eval().to(device)
cast_type = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}[precision]
cldm.cast_dtype(cast_type)

# 3. load noise schedule
diffusion: Diffusion = instantiate_from_config(
    OmegaConf.load("configs/inference/diffusion_v2.1.yaml")
)
diffusion.to(device)

# 4. load captioner
if captioner == "none":
    captioner = EmptyCaptioner(device)
elif captioner == "llava":
    captioner = LLaVACaptioner(device, llava_bit)
else:
    captioner = RAMCaptioner(device)

error_image = np.array(Image.open("assets/gradio_error_img.png"))


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
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
    )

    for batch in dataset.image_paths:
        lq = input_image
        # Prepare prompt
        caption = captioner(lq)
        print(caption)
        pos_prompt = ", ".join([text for text in [caption, positive_prompt] if text])
        neg_prompt = negative_prompt


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
