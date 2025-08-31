import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import (
    SwinIRPipeline,
    SCUNetPipeline,
    RestormerPipeline,
)
from ..model import SwinIR, SCUNet
import torch 
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from basicsr.models.archs.restormer_arch import Restormer

class BIDInferenceLoop(InferenceLoop):
    def load_cleaner(self) -> None:
        yaml_file = 'configs/inference/GaussianColorDenoising_Restormer.yml'
        weights = "/home/ziyiliu/pythonprojects/Restormer/experiments/GaussianColorDenoising_Restormer/models/net_g_92000.pth"
        x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
        self.cleaner: Restormer = Restormer(**x['network_g'])
        checkpoint = torch.load(weights)
        self.cleaner.load_state_dict(checkpoint['params'])
        print("===>Testing using weights: ",weights)
        print("------------------------------------------------")
        self.cleaner.eval().to(self.args.device)

    # def load_cleaner(self) -> None:
    #     if self.args.version == "v1":
    #         config = "configs/inference/swinir.yaml"
    #         weight = MODELS["swinir_general"]
    #     elif self.args.version == "v2":
    #         config = "configs/inference/scunet.yaml"
    #         weight = MODELS["scunet_psnr"]
    #     else:
    #         config = "configs/inference/swinir.yaml"
    #         weight = MODELS["swinir_realesrgan"]
    #     self.cleaner: SCUNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
    #     if self.args.cleaner_ckpt is None:
    #         model_weight = load_model_from_url(weight)
    #     else:
    #         print(f'load weight from {self.args.cleaner_ckpt}')
    #         model_weight = torch.load(self.args.cleaner_ckpt)
    #     self.cleaner.load_state_dict(model_weight, strict=True)
    #     self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        if self.args.version == "v1" or self.args.version == "v2.1":
            # pipeline_class = RestormerPipeline
            pipeline_class = SwinIRPipeline
        else:
            pipeline_class = SCUNetPipeline
        self.pipeline = pipeline_class(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
