# SR Mobile PyTorch

An unofficial PyTorch port (and subsequent modification) of [NJU-Jet/SR_Mobile_Quantization](https://github.com/NJU-Jet/SR_Mobile_Quantization).

## Installation

```bash
git clone https://github.com/w11wo/sr_mobile_pytorch.git
cd sr_mobile_pytorch
pip install .
```

## Pre-processing

To begin with, modify file paths, config, and training arguments in `sr_mobile_pytorch/config`. Then, launch the preprocessing script to group images into train and validation sets.

```bash
sh scripts/preprocess.sh
```

The process involves finding suitable images for validation, that is, images whose dimensions are exact multiples of 4; i.e. <img src="https://render.githubusercontent.com/render/math?math={\color{white}H \equiv 0 \mod 4}#gh-dark-mode-only"> and <img src="https://render.githubusercontent.com/render/math?math={\color{white}W \equiv 0 \mod 4}#gh-dark-mode-only">. This is to facilitate the PSNR metric calculation.

## Pre-training

We follow a slightly different pre-training regime: incorporating content loss together with pixel-wise L1 loss. This is heavily inspired by fast.ai's super resolution training.

```bash
sh scripts/pretrain.sh
```

## Fine-tuning

You can optionally fine-tune the model in a SRGAN-like setting, incorporating additional generator-discriminator loss terms.

```bash
sh scripts/finetune.sh
```

## Contributors

<a href="https://github.com/w11wo/sr_mobile_pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=w11wo/sr_mobile_pytorch" />
</a>
