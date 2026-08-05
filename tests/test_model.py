import torch
from torch import nn

from incremental_blood_cell.model import build_resnet18, expand_classifier


def test_builds_resnet18() -> None:
    model = build_resnet18(num_classes=4)

    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 4

    model.eval()

    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 4)


def test_expands_classifier() -> None:
    model = build_resnet18(num_classes=4)

    old_weight = model.fc.weight.detach().clone()
    old_bias = model.fc.bias.detach().clone()

    expand_classifier(model, num_classes=6)

    assert model.fc.out_features == 6
    assert torch.equal(model.fc.weight[:4], old_weight)
    assert torch.equal(model.fc.bias[:4], old_bias)
