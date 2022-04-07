import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)
    model = ResNet18(num_classes=1)
    output = model(x)
    print(sum(p.numel() for p in model.parameters()))
    print(output.shape)
