import numpy as np
import cv2
import onnxruntime
from glob import glob
import os
from tqdm.auto import tqdm


def pre_process(img: np.array) -> np.array:
    # H, W, C -> C, H, W
    img = np.transpose(img[:, :, 0:3], (2, 0, 1))
    # C, H, W -> 1, C, H, W
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def post_process(img: np.array) -> np.array:
    # 1, C, H, W -> C, H, W
    img = np.squeeze(img)
    # C, H, W -> H, W, C
    img = np.transpose(img, (1, 2, 0))
    return img


def save(img: np.array, save_name: str) -> None:
    cv2.imwrite(save_name, img)


def inference(model_path: str, img_array: np.array) -> np.array:
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: img_array}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs[0]


def main():
    model_path = "./experiments/generator_minecraft_x4/model.ort"
    path = "./assets"
    save_path = os.path.join(path, "output")
    os.makedirs(save_path, exist_ok=True)
    images = glob(f"{path}/*.png") + glob(f"{path}/*.jpeg") + glob(f"{path}/*.jpg")

    for image_path in tqdm(images):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        filename = os.path.basename(image_path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            alpha = img[:, :, 3]  # GRAY
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # BGR
            alpha_output = post_process(
                inference(model_path, pre_process(alpha))
            )  # BGR
            alpha_output = cv2.cvtColor(alpha_output, cv2.COLOR_BGR2GRAY)  # GRAY

            img = img[:, :, 0:3]  # BGR
            image_output = post_process(inference(model_path, pre_process(img)))  # BGR
            output_img = cv2.cvtColor(image_output, cv2.COLOR_BGR2BGRA)  # BGRA
            output_img[:, :, 3] = alpha_output
            save(output_img, f"{save_path}/{filename}")
        elif img.shape[2] == 3:
            image_output = post_process(inference(model_path, pre_process(img)))  # BGR
            save(image_output, f"{save_path}/{filename}")


if __name__ == "__main__":
    main()

