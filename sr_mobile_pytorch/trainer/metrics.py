import numpy as np
import math


def calculate_psnr(y, y_target):
    _, h, w, _ = y.shape
    y = np.clip(np.round(y), 0, 255).astype(np.float32)
    y_target = np.clip(np.round(y_target), 0, 255).astype(np.float32)

    # crop 1
    y_cropped = y[:, :, 1 : h - 1, 1 : w - 1]
    y_target_cropped = y_target[:, :, 1 : h - 1, 1 : w - 1]

    mse = np.mean((y_cropped - y_target_cropped) ** 2)
    if mse == 0:
        return 100
    return 20.0 * math.log10(255.0 / math.sqrt(mse))
