import torch
import torch.nn as nn
from torchvision.models import resnet18
import copy

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
        self.layers = [
            "layer1.0.relu",
            "layer1.1.relu",
            "layer2.0.relu",
            "layer2.1.relu",
            "layer3.0.relu",
            "layer3.1.relu",
            "layer4.0.relu",
            "layer4.1.relu",
        ]
        self._features = {layer: torch.empty(0) for layer in self.layers}
        for layer_id in self.layers:
            layer = dict(self.model.named_modules())[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def load_resnet_feature_extractor(self, model_path, device):
        resnet = resnet18(pretrained=False)
        weights = torch.load(model_path, map_location=device)
        state_dict = weights["state_dict"]
        for k in list(state_dict.keys()):
            if k.startswith("backbone.") and not k.startswith("backbone.fc"):
                state_dict[k[len("backbone.") :]] = state_dict[k]
            del state_dict[k]
        resnet.load_state_dict(state_dict, strict=False)
        for param in resnet.parameters():
            param.requires_grad = False
        return resnet.eval().to(device)

    def forward(self, hr, sr):
        hr, sr = hr / 255.0, sr / 255.0
        self.model(sr)
        sr_features = copy.deepcopy(self._features)
        self.model(hr)
        hr_features = copy.deepcopy(self._features)
        loss = torch.tensor(0.0)
        for layer in self.layers:
            loss += self.mae_loss(sr_features[layer], hr_features[layer])
        return loss


class GANLoss:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(self, sr_out):
        return self.bce_loss(sr_out, torch.ones_like(sr_out))

    def discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.bce_loss(hr_out, torch.ones_like(hr_out))
        sr_loss = self.bce_loss(sr_out, torch.zeros_like(sr_out))
        return hr_loss + sr_loss
