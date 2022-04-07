# SR Mobile PyTorch

An unofficial PyTorch port of [NJU-Jet/SR_Mobile_Quantization](https://github.com/NJU-Jet/SR_Mobile_Quantization).

## Installation

```bash
git clone https://github.com/w11wo/sr_mobile_pytorch.git
cd sr_mobile_pytorch
pip install .
```

## Training

Modify file paths, config, and training arguments in `sr_mobile_pytorch/config`.

```bash
bash scripts/preprocess.sh
bash scripts/pretrain.sh
bash scripts/finetune.sh
```

## Contributors

<a href="https://github.com/w11wo/sr_mobile_pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=w11wo/sr_mobile_pytorch" />
</a>
