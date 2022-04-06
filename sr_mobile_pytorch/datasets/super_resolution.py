import random
import numpy as np
import cv2


class SuperResolutionDataset:
    def __init__(self, df, scale=4, patch_size=64, train=True, flip=True, rotate=True):
        self.df = df
        self.train = train
        self.scale = scale
        self.patch_size = patch_size
        self.flip = flip
        self.rotate = rotate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        lr_path, hr_path = self.df.iloc[idx]["lr"], self.df.iloc[idx]["hr"]
        lr, hr = self.read_img(lr_path), self.read_img(hr_path)
        if self.train:
            lr, hr = self.patch_image(lr, hr)
            lr, hr = self.augment(lr, hr)
        return (lr).astype(np.float32), (hr).astype(np.float32)

    def patch_image(self, lr, hr):
        lr_h, lr_w = lr.shape[1:]

        lr_x = random.randint(0, max(lr_w - self.patch_size - 1, 0))
        lr_y = random.randint(0, max(lr_h - self.patch_size - 1, 0))
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale

        lr_patch = lr[:, lr_y : lr_y + self.patch_size, lr_x : lr_x + self.patch_size]
        hr_patch = hr[
            :,
            hr_y : hr_y + self.patch_size * self.scale,
            hr_x : hr_x + self.patch_size * self.scale,
        ]

        return lr_patch, hr_patch

    def augment(self, lr, hr):
        hflip = self.flip and random.random() < 0.5
        vflip = self.flip and random.random() < 0.5
        rot90 = self.rotate and random.random() < 0.5

        if hflip:
            lr = np.ascontiguousarray(lr[:, :, ::-1])
            hr = np.ascontiguousarray(hr[:, :, ::-1])
        if vflip:
            lr = np.ascontiguousarray(lr[:, ::-1, :])
            hr = np.ascontiguousarray(hr[:, ::-1, :])
        if rot90:
            lr = lr.transpose(0, 2, 1)
            hr = hr.transpose(0, 2, 1)

        return lr, hr

    def read_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, axes=(2, 0, 1))
        img = img.astype(np.float32)
        return img
