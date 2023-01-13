import torch.nn as nn

from .resnet_blocks import Block, Bottleneck


class ResNet(nn.Module):
    def __init__(self, res_block, layer_list, num_classes, num_channels):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(res_block, layer_list[0], planes=64)
        self.layer2 = self._make_layer(res_block, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(res_block, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(res_block, layer_list[3], planes=512, stride=2)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def build_resnet18(num_classes, channels=1):
    return ResNet(Block, [2, 2, 2, 2], num_classes, channels)


def build_resnet34(num_classes, channels=1):
    return ResNet(Block, [3, 4, 6, 3], num_classes, channels)


def build_resnet50(num_classes, channels=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def build_resnet101(num_classes, channels=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def build_resnet152(num_classes, channels=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)
