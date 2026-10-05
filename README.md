The code of A Unified Low-level Foundation Model for Enhancing Pathology Image Quality.

## Prerequisites

Python: 3.8 or higher

PyTorch: 1.12.0 or higher with CUDA support

GPU: NVIDIA GPU with ≥8GB VRAM (recommended)

Dependencies: See requirements.txt

## Image restoration 

Some samples: https://drive.google.com/file/d/13BSlv-FSk8FNW3YHtOksaKj92MxiX1yP/view?usp=sharing

Please download the following pre-trained weights for image restoration:

Autoencoder for coarse restoration: [lpfm_restore_stage1](https://drive.google.com/file/d/1NbmHW8s3Q0H_58asEHxSpSmX3JKM3fqH/view?usp=sharing)

Diffusion network for image refinement: [lpfm_restore_stage2](https://drive.google.com/file/d/148a3V3NU_uhUbWlDZ9QG2JEbcdy9Ve_Q/view?usp=sharing)

Basical ControlNet: [diffusion](https://drive.google.com/file/d/1Frk8Anr4hWvW1qtv57Mdqh9Jsl3Bq_U2/view?usp=sharing)

Put the degraded pathology images in the path: 'args.input'. 
We can get the restored pathology images in the path: 'args.output'

Directory structure after setup:

LPFM/

├── weights/

│   ├── lpfm_restore_stage1.pth

│   └── lpfm_restore_stage2.pt

│   └── v2-1_512-ema-pruned.ckpt

├── samples/          # Input images

├── outputs/          # Restored images

├── inference.py

└── requirements.txt


Run the code below to get the restored images.

```python
python inference.py 
  --upscale 1 \
  --version v1 \
  --sampler spaced \
  --steps 50 \
  --captioner none \
  --pos_prompt '' \
  --neg_prompt 'low quality, blurry, low-resolution, noisy' \
  --cfg_scale 1 \
  --input ./samples \
  --output ./outputs \
  --cleaner_ckpt ./weights/lpfm_restore_stage1.pth \
  --sd_ckpt ./v2-1_512-ema-pruned.ckpt \
  --cldm_ckpt ./weights/lpfm_restore_stage2.pt \
  --device cuda \
  --precision fp32

```

## Image translation (virtual staining)

Please download the HEMIT dataset: [HEMIT](https://github.com/BianChang/HEMIT-DATASET)

Please download the following pre-trained weights for image translation:

Autoencoder for coarse restoration: [stage1_vs](https://drive.google.com/file/d/1cuLF0b9vVKSXkCCHDFNAklQAuGzOU-qZ/view?usp=sharing)

Diffusion network for image refinement: [stage2_vs](https://drive.google.com/file/d/131Hf1RXPRJqApIDIwdZ36dn_XoxILRvQ/view?usp=sharing)

Basical ControlNet: [diffusion](https://drive.google.com/file/d/1Frk8Anr4hWvW1qtv57Mdqh9Jsl3Bq_U2/view?usp=sharing)

Directory structure after setup:

LPFM/

├── weights/

│   ├── stage1_vs.pth

│   └── stage2_vs.pt

│   └── v2-1_512-ema-pruned.ckpt


Run the code below to get the virtual staining images.

```python
accelerate launch test_stage2.py --config configs/train/test_stage2_hemit.yaml

```
