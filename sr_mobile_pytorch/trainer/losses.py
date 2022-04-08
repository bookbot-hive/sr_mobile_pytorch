import torch
import torch.nn as nn
from torchvision.models import resnet18

from sr_mobile_pytorch.trainer.utils import imagenet_normalize


class ContentLossVGG(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mae_loss = nn.L1Loss()
        self.vgg = torch.hub.load("pytorch/vision:v0.10.0", "vgg19", pretrained=True)
        self.model = nn.Sequential(*[self.vgg.features[i] for i in range(36)]).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.to(device)

    def forward(self, hr, sr):
        sr = imagenet_normalize(sr)
        hr = imagenet_normalize(hr)
        sr_features = self.model(sr)
        hr_features = self.model(hr)
        return self.mae_loss(hr_features, sr_features)


class ContentLossResNetSimCLR(nn.Module):
    def __init__(self, feature_extactor_path, device):
        super().__init__()
        self.device = device
        self.mae_loss = nn.L1Loss()
        self.model = self.load_resnet_feature_extractor(feature_extactor_path, device)

    def load_resnet_feature_extractor(self, model_path, device):
        resnet = resnet18(pretrained=False)
        weights = torch.load(model_path, map_location=device)
        state_dict = weights["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("backbone.") and not k.startswith("backbone.fc"):
                state_dict[k[len("backbone.") :]] = state_dict[k]
            del state_dict[k]
        resnet.load_state_dict(state_dict, strict=False)
        model = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        for param in model.parameters():
            param.requires_grad = False
        return model.eval().to(device)

    def forward(self, hr, sr):
        hr, sr = hr / 255.0, sr / 255.0
        sr_features = self.model(sr)
        hr_features = self.model(hr)
        return self.mae_loss(hr_features, sr_features)


class GANLoss:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(self, sr_out):
        return self.bce_loss(sr_out, torch.ones_like(sr_out))

    def discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.bce_loss(hr_out, torch.ones_like(hr_out))
        sr_loss = self.bce_loss(sr_out, torch.zeros_like(sr_out))
        return hr_loss + sr_loss
