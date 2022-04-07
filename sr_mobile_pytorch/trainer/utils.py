import random
import os
import torch
import numpy as np
import logging
import json
from typing import Tuple, Dict, Any


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


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger = logging.getLogger("sr_mobile_pytorch")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
