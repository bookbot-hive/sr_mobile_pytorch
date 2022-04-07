import random
import os
import torch
import numpy as np
import logging
import json
from typing import Tuple, Dict, Any
import torchvision.transforms as transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(path, "r") as f:
        config = json.load(f)

    return config["model_args"], config["training_args"]


def imagenet_normalize(t: torch.Tensor):
    t = t / 255.0
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return normalize(t)


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger = logging.getLogger("sr_mobile_pytorch")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
