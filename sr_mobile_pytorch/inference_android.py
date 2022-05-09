import numpy as np
import cv2
import onnxruntime
from glob import glob
import os
from tqdm.auto import tqdm


def tensor_to_bitmap(x: np.array) -> np.array:
    x = np.transpose(x, (0, 2, 3, 1)).astype(np.int32)
    mask = np.array([16, 8, 0], dtype=np.int32)
    ls = np.left_shift(x, mask)
    bitmap = (np.sum(ls, axis=-1).flatten().astype(np.int32) - 16777216).astype(
        np.float32
    )
    return bitmap


def bitmap_to_tensor(bitmap: np.array, height: int, width: int) -> np.array:
    x = np.repeat(bitmap.reshape((1, height, width, 1)), 3, axis=3).astype(np.int32)
    mask = np.array([16, 8, 0], dtype=np.int32)
    rs = np.right_shift(x, mask).astype(np.uint8)
    return rs


def pre_process(img: np.array, height, width) -> np.array:
    # H, W, C -> C, H, W
    img = np.transpose(img[:, :, 0:3], (2, 0, 1))
    # C, H, W -> 1, C, H, W
    img = np.expand_dims(img, axis=0).astype(np.float32)
    bitmap = tensor_to_bitmap(img).reshape(1, 1, height, width)
    return bitmap


def post_process(img: np.array, height: int, width: int) -> np.array:
    # 1, C, H, W -> C, H, W
    img = bitmap_to_tensor(img, height, width)
    img = cv2.cvtColor(np.squeeze(img), cv2.COLOR_RGB2BGR)
    return img


def save(img: np.array, save_name: str) -> None:
    cv2.imwrite(save_name, img)


def inference(model_path: str, img_array: np.array) -> np.array:
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {"input": img_array}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs[0]


def main():
    model_path = "./experiments/generator_v4_channel_last_android/model_channel_last_android.onnx"
    path = "./assets"
    save_path = os.path.join(path, "output")
    os.makedirs(save_path, exist_ok=True)
    images = glob(f"{path}/*.png") + glob(f"{path}/*.jpeg") + glob(f"{path}/*.jpg")

    for image_path in tqdm(images):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(image_path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        height, width, _ = img.shape
        print(img.shape)
        SCALE = 4
        input_height = np.array(height).astype(np.int64)
        input_width = np.array(width).astype(np.int64)
        output_height = np.array(height * SCALE).astype(np.int64)
        output_width = np.array(width * SCALE).astype(np.int64)

        if img.shape[2] == 4:
            alpha = img[:, :, 3]  # GRAY
            alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # BGR
            alpha_output = post_process(
                inference(model_path, pre_process(alpha, input_height, input_width)),
                output_height,
                output_width,
            )  # BGR
            alpha_output = cv2.cvtColor(alpha_output, cv2.COLOR_BGR2GRAY)  # GRAY

            img = img[:, :, 0:3]  # BGR
            image_output = post_process(
                inference(model_path, pre_process(img, input_height, input_width)),
                output_height,
                output_width,
            )  # BGR
            output_img = cv2.cvtColor(image_output, cv2.COLOR_BGR2BGRA)  # BGRA
            output_img[:, :, 3] = alpha_output
            save(output_img, f"{save_path}/{filename}")
        elif img.shape[2] == 3:
            image_output = post_process(
                inference(model_path, pre_process(img, input_height, input_width)),
                output_height,
                output_width,
            )  # BGR
            save(image_output, f"{save_path}/{filename}")


if __name__ == "__main__":
    main()

