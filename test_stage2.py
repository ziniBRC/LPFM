import os
import numpy as np 
from argparse import ArgumentParser
from PIL import Image
import copy
import cv2
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, wavelet_reconstruction, low_freq_wavelet_reconstruction
from diffbir.sampler import SpacedSampler

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # Setup optimizer:
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training:
    cldm.eval().to(device)
    swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    # template_path = "/mnt/data/ziyiliu/virtual_staining/he2pas_uniform/org_test/a_handled_A-05/1_15/he_75616_11840_256.png"
    # template = Image.open(template_path).convert("RGB")
    # template = np.array(template)
    # template = (template[..., ::-1] / 255.0).astype(np.float32)
    # template = template[..., ::-1].astype(np.float32)
    # template = torch.from_numpy(template)
    # template = rearrange(template[None, :], "b h w c -> b c h w").contiguous().float()
    # template = template.cuda()

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt, lq_name = batch
            
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                cond = pure_cldm.prepare_condition(clean, prompt)
                # noise augmentation
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug)

            N = 1
            log_clean = clean[:N]
            log_cond = {k: v[:N] for k, v in cond.items()}
            log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
            log_gt, log_lq = gt[:N], lq[:N]
            log_prompt = prompt[:N]

            with torch.no_grad():
                z = sampler.sample(
                    model=cldm,
                    device=device,
                    steps=50,
                    x_size=(len(log_gt), *z_0.shape[1:]),
                    cond=log_cond,
                    uncond=None,
                    cfg_scale=1.0,
                    progress=accelerator.is_main_process,
                )
                samples = (pure_cldm.vae_decode(z) + 1) / 2 * 255 
                gt = (log_gt + 1) / 2 * 255
                log_clean = log_clean * 255
                log_lq = log_lq * 255
                condition_decoded = (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2 * 255
                condition_aug_decoded = (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2 * 255

                samples = F.interpolate(
                wavelet_reconstruction(samples, gt),
                size=(256,256),
                mode="bicubic",
                antialias=True,
                )

                samples = samples.clip(0, 255)
                gt = gt.clip(0, 255)
                log_clean = log_clean.clip(0, 255)
                log_lq = log_lq.clip(0, 255)
                condition_decoded = condition_decoded.clip(0, 255)
                condition_aug_decoded = condition_aug_decoded.clip(0, 255)

                # save_dir = '/mnt/data/ziyiliu/test_images/hemit/af2he_diff_test_uniform'
                # save_dir = '/mnt/data/ziyiliu/test_images/he2pas_degraded/diff_test_uniform'
                # save_dir = '/mnt/data/ziyiliu/test_images/he2pas/af2he_diff_test_uniform'
                # gt_dir = '/mnt/data/ziyiliu/test_images/hemit/gt'
                refinement_dir = '/mnt/data/ziyiliu/test_images/he2pas/diff_refinement'
                autoencoder_dir = '/mnt/data/ziyiliu/test_images/he2pas/diff_autoencoder'
                os.makedirs(refinement_dir, exist_ok=True)
                os.makedirs(autoencoder_dir, exist_ok=True)
                Results = Image.fromarray((rearrange(samples[0, :].detach().cpu().numpy(), 'c h w -> h w c')).astype(np.uint8))
                # Results.save('samples.png')
                print(f'save {os.path.join(refinement_dir, lq_name[0])}')
                Results.save(os.path.join(refinement_dir, lq_name[0]))
                Results = Image.fromarray((rearrange(gt[0, :].detach().cpu().numpy(), 'c h w -> h w c')).astype(np.uint8))
                # Results.save(os.path.join(gt_dir, lq_name[0]))
                # Results.save('gt.png')
                # Results = Image.fromarray((rearrange(log_lq[0, :].detach().cpu().numpy(), 'c h w -> h w c')).astype(np.uint8))
                # Results.save('lq.png')
                Results = Image.fromarray((rearrange(log_clean[0, :].detach().cpu().numpy(), 'c h w -> h w c')).astype(np.uint8))
                # Results.save('condition.png')
                print(f'save {os.path.join(autoencoder_dir, lq_name[0])}')
                Results.save(os.path.join(autoencoder_dir, lq_name[0]))
                # Results = Image.fromarray((rearrange(condition_decoded[0, :].detach().cpu().numpy(), 'c h w -> h w c')).astype(np.uint8))
                # Results.save('condition_decoded.png')
                print('pass')

        
        accelerator.wait_for_everyone()
        if global_step == max_steps:
            break

        pbar.close()
        epoch += 1

        # only run one epoch 
        if epoch == 1:
            break

        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
