import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):

    def __init__(self, truncation, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        # self.features = features
        self.truncation = truncation
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 =  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256) 
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256) 
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512) 
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(512) 
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(512) 
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512) 
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.bn1(x0)
        x2 = self.relu(x1)
        x3 = self.conv2(x2)
        x4 = self.bn2(x3)
        x5 = self.relu(x4)
        x6 = self.maxpool(x5)
        
        x7 = self.conv3(x6)
        x8 = self.bn3(x7)
        x9 = self.relu(x8)
        x10 = self.conv4(x9)
        x11 = self.bn4(x10)
        x12 = self.relu(x11)
        x13 = self.maxpool(x12)

        x14 = self.conv5(x13)
        x15 = self.bn5(x14)
        x16 = self.relu(x15)
        x17 = self.conv6(x16)
        x18 = self.bn6(x17)
        x19 = self.relu(x18)
        x20 = self.conv7(x19)
        x21 = self.bn7(x20)
        x22 = self.relu(x21)
        x23 = self.maxpool(x22)

        x24 = self.conv8(x23)
        x25 = self.bn8(x24)
        x26 = self.relu(x25)
        x27 = self.conv9(x26)
        x28 = self.bn9(x27)
        x29 = self.relu(x28)
        x30 = self.conv10(x29)
        x31 = self.bn10(x30)
        x32 = self.relu(x31)
        x33 = self.maxpool(x32)

        x34 = self.conv11(x33)
        x35 = self.bn11(x34)
        x36 = self.relu(x35)
        x37 = self.conv12(x36)
        x38 = self.bn12(x37)
        x39 = self.relu(x38)
        x40 = self.conv13(x39)
        x41 = self.bn13(x40)
        x42 = self.relu(x41)
        x43 = self.maxpool(x42)


        x = self.avgpool(x43)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.truncation == "layer0":
            return x2, x
        elif self.truncation == "layer1":
            return x6, x
        elif self.truncation == "layer2":
            return x9, x
        elif self.truncation == "layer3":
            return x13, x
        elif self.truncation == "layer4":
            return x16, x
        elif self.truncation == "layer5":
            return x19, x
        elif self.truncation == "layer6":
            return x23, x
        elif self.truncation == "layer7":
            return x26, x
        elif self.truncation == "layer8":
            return x29, x
        elif self.truncation == "layer9":
            return x33, x
        elif self.truncation == "layer10":
            return x36, x
        elif self.truncation == "layer11":
            return x39, x
        elif self.truncation == "layer12":
            return x43, x
        else:
            return x16, x19, x39, x43, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _vgg(arch, cfg, batch_norm, pretrained, progress, truncation, **kwargs):
    # if pretrained:
    #     kwargs['init_weights'] = False
    model = VGG(truncation, **kwargs)
    return model



def vgg16_trunc(pretrained=False, progress=True, truncation=None, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, truncation, **kwargs)