import torch
from torch import nn


class UpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(UpscaleConvBlock, self).__init__()
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nn.ReLU()
        )

    def forward(self, x):
        return self.upscale(x)


class AnchorBasedPlainNet(nn.Module):
    def __init__(
        self, scale=4, in_channels=3, num_feature=28, num_encoder=4, out_channels=3
    ):
        super(AnchorBasedPlainNet, self).__init__()
        self.scale = scale
        self.num_feature = num_feature
        self.num_encoder = num_encoder
        self.upsample_func = lambda x_list: torch.concat(x_list, dim=1)
        self.activation = nn.ReLU()

        # feature extraction blocks
        self.shallow_feature_extraction = UpscaleConvBlock(
            3, num_feature, 3, (1, 1), (1, 1)
        )
        self.deep_feature_extraction = nn.Sequential(
            *[
                UpscaleConvBlock(num_feature, num_feature, 3, (1, 1), (1, 1))
                for _ in range(self.num_encoder)
            ]
        )
        output_kernel = out_channels * (self.scale ** 2)
        self.last_feature_extraction = UpscaleConvBlock(
            num_feature, output_kernel, 3, (1, 1), (1, 1)
        )

        # transition
        self.transition = nn.Conv2d(output_kernel, output_kernel, 3, (1, 1), (1, 1))

        # pixel shuffle block
        self.depth_to_space = nn.PixelShuffle(self.scale)

    def forward(self, x):
        # identity
        upsample_inp = self.upsample_func([x] * self.scale ** 2)

        # feature extraction
        x = self.shallow_feature_extraction(x)
        x = self.deep_feature_extraction(x)
        x = self.last_feature_extraction(x)

        # transition
        x = self.transition(x)

        # skip connection
        x = x + upsample_inp

        # pixel-shuffle
        out = self.depth_to_space(x)

        # clamp
        out = torch.clamp(out, min=0.0, max=255.0)
        return out
