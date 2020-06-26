import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, truncation, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.truncation = truncation

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_0 = self._make_layer(block, 64, layers[0])
        self.layer1_1 = self._make_layer(block, 64, layers[1])
        self.layer1_2 = self._make_layer(block, 64, layers[2])

        self.layer2_0 = self._make_layer(block, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2_1 = self._make_layer(block, 128, layers[4], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2_2 = self._make_layer(block, 128, layers[5], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2_3 = self._make_layer(block, 128, layers[6], stride=1,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3_0 = self._make_layer(block, 256, layers[7], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3_1 = self._make_layer(block, 256, layers[8], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3_2 = self._make_layer(block, 256, layers[9], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3_3 = self._make_layer(block, 256, layers[10], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3_4 = self._make_layer(block, 256, layers[11], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3_5 = self._make_layer(block, 256, layers[12], stride=1,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4_0 = self._make_layer(block, 512, layers[13], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.layer4_1 = self._make_layer(block, 512, layers[14], stride=1,
                                       dilate=replace_stride_with_dilation[2])     
        self.layer4_2 = self._make_layer(block, 512, layers[15], stride=1,
                                       dilate=replace_stride_with_dilation[2])     

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  
        x = self.relu(x) 
        x0 = self.maxpool(x)

        x1 = self.layer1_0(x0) 
        x2 = self.layer1_1(x1)
        x3 = self.layer1_2(x2) 

        x4 = self.layer2_0(x3) 
        x5 = self.layer2_1(x4)
        x6 = self.layer2_2(x5)
        x7 = self.layer2_3(x6)

        x8 = self.layer3_0(x7) 
        x9 = self.layer3_1(x8)
        x10 = self.layer3_2(x9)
        x11 = self.layer3_3(x10)
        x12 = self.layer3_4(x11)
        x13 = self.layer3_5(x12)

        x14 = self.layer4_0(x13) 
        x15 = self.layer4_1(x14)
        x16 = self.layer4_2(x15)

        x = self.avgpool(x16) 
        x = x.reshape(x.size(0), -1) 
        x = self.fc(x)

        if self.truncation == 'layer0':
            return x0, x
        elif self.truncation == 'layer1':
            return x1, x
        elif self.truncation == 'layer2':
            return x2, x
        elif self.truncation == 'layer3':
            return x3, x
        elif self.truncation == 'layer4':
            return x4, x
        elif self.truncation == 'layer5':
            return x5, x
        elif self.truncation == 'layer6':
            return x6, x
        elif self.truncation == 'layer7':
            return x7, x
        elif self.truncation == 'layer8':
            return x8, x
        elif self.truncation == 'layer9':
            return x9, x
        elif self.truncation == 'layer10':
            return x10, x
        elif self.truncation == 'layer11':
            return x11, x
        elif self.truncation == 'layer12':
            return x12, x
        elif self.truncation == 'layer13':
            return x13, x
        elif self.truncation == 'layer14':
            return x14, x
        elif self.truncation == 'layer15':
            return x15, x
        elif self.truncation == 'layer16':
            return x16, x
        else:
            return x7, x16, x


def _resnet(arch, block, layers, pretrained, progress, truncation, num_classes, **kwargs):
    # if truncation in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
    #     model = ResNet(block, layers, truncation, num_classes, **kwargs)
    #     if pretrained:
    #         state_dict = load_state_dict_from_url(model_urls[arch],
    #                                             progress=progress)
    #         model.load_state_dict(state_dict)
    #     return model
    # else:
    model = ResNet(block, layers, truncation, num_classes, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                         progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet50_trunc(pretrained=False, progress=True, truncation=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], pretrained, progress, truncation, num_classes,
                **kwargs)
    # return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, truncation, num_classes,
    #                **kwargs)
