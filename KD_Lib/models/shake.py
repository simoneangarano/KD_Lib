import torch
from torch import nn
from torch.nn import functional as F

class ShakeHead(nn.Module):
    """Convolutional regression for FitNet (feature-map layer)"""

    def __init__(self, feat_t):
        super(ShakeHead, self).__init__()
 
        self.fuse1 = conv_bn(feat_t[1].shape[1], feat_t[2].shape[1], 2)
        self.fuse2 = conv_1x1_bn(feat_t[2].shape[1], feat_t[2].shape[1])
        self.fuse3 = conv_bn(feat_t[2].shape[1], feat_t[3].shape[1], 2)
        self.fuse4 = conv_1x1_bn(feat_t[3].shape[1], feat_t[3].shape[1])
        self.fuse5 = conv_bn(feat_t[3].shape[1], feat_t[3].shape[1], 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(feat_t[3].shape[1], 100)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, f, weight, bias):
        x = self.fuse5(self.fuse3(self.fuse1(f[0]) + self.fuse2(f[1])) + self.fuse4(f[2]))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.linear(x, weight, bias)
        # x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        # nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))

def conv_1x1_bn(num_input_channels, num_mid_channel):
    return nn.Sequential(
        conv1x1(num_input_channels, num_mid_channel),
        nn.BatchNorm2d(num_mid_channel),
        nn.ReLU(inplace=True),
        # conv3x3(num_mid_channel, num_mid_channel),
        # nn.BatchNorm2d(num_mid_channel),
        # nn.ReLU(inplace=True),
        # conv1x1(num_mid_channel, num_mid_channel),
        )
