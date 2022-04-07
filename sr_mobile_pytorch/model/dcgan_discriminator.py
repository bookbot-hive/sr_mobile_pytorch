from pyexpat import features
import torch
import torch.nn as nn


class DCGANDiscriminator(nn.Module):
    # Adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/model.py
    def __init__(self, features_d):
        super(DCGANDiscriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            self._block(features_d * 8, features_d * 16, 4, 2, 1),
            self._block(features_d * 16, features_d * 32, 4, 2, 1),
            nn.Conv2d(features_d * 32, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)
    model = DCGANDiscriminator(features_d=4)
    output = model(x)
    print(sum(p.numel() for p in model.parameters()))
    print(output.shape)
