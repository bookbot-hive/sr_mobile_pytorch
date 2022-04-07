import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ContentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.vgg = torch.hub.load(
            "pytorch/vision:v0.10.0", "vgg19", pretrained=True
        ).to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.model = self.vgg.features.eval()

    def preprocess_input(self, x):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return normalize(x / 255.0)

    def forward(self, hr, sr):
        sr = self.preprocess_input(sr)
        hr = self.preprocess_input(hr)
        sr_features = self.model(sr) / 12.75
        hr_features = self.model(hr) / 12.75
        return self.mse_loss(hr_features, sr_features)


class GANLoss:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(self, sr_out):
        return self.bce_loss(torch.ones_like(sr_out), sr_out)

    def discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.bce_loss(torch.ones_like(hr_out), hr_out)
        sr_loss = self.bce_loss(torch.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
