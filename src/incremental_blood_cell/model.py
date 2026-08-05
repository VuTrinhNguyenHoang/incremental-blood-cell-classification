import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet


def build_resnet18(num_classes: int) -> ResNet:
    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    return model


def expand_classifier(model: ResNet, num_classes: int) -> None:
    old_head = model.fc

    new_head = nn.Linear(in_features=old_head.in_features, out_features=num_classes)
    new_head = new_head.to(device=old_head.weight.device, dtype=old_head.weight.dtype)

    with torch.no_grad():
        new_head.weight[: old_head.out_features].copy_(old_head.weight)
        new_head.bias[: old_head.out_features].copy_(old_head.bias)

    model.fc = new_head
