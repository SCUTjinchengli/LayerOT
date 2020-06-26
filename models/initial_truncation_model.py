import torch
import torch.nn as nn
import logging
import pdb


class AdditionalLayerTemplet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdditionalLayerTemplet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()
        linear_params = list(self.fc1.parameters())
        linear_params[0].data.normal_(0, 0.01)
        linear_params[0].data.fill_(0)
        linear_params_2 = list(self.fc2.parameters())
        linear_params_2[0].data.normal_(0, 0.01)
        linear_params_2[0].data.fill_(0)
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.sigmoid(x)


def get_additional_layer(resnet_model, truncation, num_classes):
    if resnet_model == "Resnet18":
        if truncation == 'layer0':
            templet = AdditionalLayerTemplet(64*56*56, num_classes)
        elif truncation == 'layer1':
            templet = AdditionalLayerTemplet(64*56*56, num_classes)
        elif truncation == 'layer2':
            templet = AdditionalLayerTemplet(64*56*56, num_classes)
        elif truncation == 'layer3':
            templet = AdditionalLayerTemplet2(128, num_classes)
        elif truncation == 'layer4':
            templet = AdditionalLayerTemplet(128*28*28, num_classes)
        elif truncation == 'layer5':
            templet = AdditionalLayerTemplet(256*14*14, num_classes)
        elif truncation == 'layer6':
            templet = AdditionalLayerTemplet(256*14*14, num_classes)
        elif truncation == 'layer7':
            templet = AdditionalLayerTemplet(512*7*7, num_classes)
        elif truncation == 'layer8':
            templet = AdditionalLayerTemplet2(512, num_classes)
        else:
            assert False
            
    elif resnet_model == "Resnet50":
        if truncation == 'layer0':
            templet = AdditionalLayerTemplet(64*56*56, num_classes)
        elif truncation == 'layer1':
            templet = AdditionalLayerTemplet(256*56*56, num_classes)
        elif truncation == 'layer2':
            templet = AdditionalLayerTemplet(256*56*56, num_classes)
        elif truncation == 'layer3':
            templet = AdditionalLayerTemplet(256*56*56, num_classes)
        elif truncation == 'layer4':
            templet = AdditionalLayerTemplet(512*28*28, num_classes)
        elif truncation == 'layer5':
            templet = AdditionalLayerTemplet(512*28*28, num_classes)
        elif truncation == 'layer6':
            templet = AdditionalLayerTemplet(512*28*28, num_classes)
        elif truncation == 'layer7':
            templet = AdditionalLayerTemplet(512*28*28, num_classes)
        elif truncation == 'layer8':
            templet = AdditionalLayerTemplet(1024*14*14, num_classes)
        elif truncation == 'layer9':
            templet = AdditionalLayerTemplet(1024*14*14, num_classes)
        elif truncation == 'layer10':
            templet = AdditionalLayerTemplet(1024*14*14, num_classes)
        elif truncation == 'layer11':
            templet = AdditionalLayerTemplet(1024*14*14, num_classes)
        elif truncation == 'layer12':
            templet = AdditionalLayerTemplet(1024*14*14, num_classes)
        elif truncation == 'layer13':
            templet = AdditionalLayerTemplet(1024*14*14, num_classes)
        elif truncation == 'layer14':
            templet = AdditionalLayerTemplet(2048*7*7, num_classes)
        elif truncation == 'layer15':
            templet = AdditionalLayerTemplet(2048, num_classes)
        elif truncation == 'layer16':
            templet = AdditionalLayerTemplet(2048, num_classes)

    elif resnet_model == "VGG16":
        if truncation == 'layer0':
            templet = AdditionalLayerTemplet(64*224*224, num_classes)
        elif truncation == 'layer1':
            templet = AdditionalLayerTemplet(64*112*112, num_classes)
        elif truncation == 'layer2':
            templet = AdditionalLayerTemplet(128*112*112, num_classes)
        elif truncation == 'layer3':
            templet = AdditionalLayerTemplet(128*56*56, num_classes)
        elif truncation == 'layer4':
            templet = AdditionalLayerTemplet(256*56*56, num_classes)
        elif truncation == 'layer5':
            templet = AdditionalLayerTemplet(256*56*56, num_classes)
        elif truncation == 'layer6':
            templet = AdditionalLayerTemplet(256*28*28, num_classes)
        elif truncation == 'layer7':
            templet = AdditionalLayerTemplet(512*28*28, num_classes)
        elif truncation == 'layer8':
            templet = AdditionalLayerTemplet(512*28*28, num_classes)
        elif truncation == 'layer9':
            templet = AdditionalLayerTemplet(512*14*14, num_classes)
        elif truncation == 'layer10':
            templet = AdditionalLayerTemplet(512*14*14, num_classes)
        elif truncation == 'layer11':
            templet = AdditionalLayerTemplet(512*14*14, num_classes)
        elif truncation == 'layer12':
            templet = AdditionalLayerTemplet(512*7*7, num_classes)

    return templet


def initialization_model_truncation(resnet_model, model_truncation, model):

    if resnet_model == "Resnet18":
        model_truncation.conv1.weight = model.conv1.weight
        model_truncation.bn1.weight = model.bn1.weight
        model_truncation.bn1.bias = model.bn1.bias
        model_truncation.bn1.running_mean = model.bn1.running_mean
        model_truncation.bn1.running_var = model.bn1.running_var

        model_truncation.layer1_0[0].conv1.weight = model.layer1[0].conv1.weight
        model_truncation.layer1_0[0].bn1.weight = model.layer1[0].bn1.weight
        model_truncation.layer1_0[0].bn1.bias = model.layer1[0].bn1.bias
        model_truncation.layer1_0[0].bn1.running_mean = model.layer1[0].bn1.running_mean
        model_truncation.layer1_0[0].bn1.running_var = model.layer1[0].bn1.running_var
        model_truncation.layer1_0[0].conv2.weight = model.layer1[0].conv2.weight
        model_truncation.layer1_0[0].bn2.weight = model.layer1[0].bn2.weight
        model_truncation.layer1_0[0].bn2.bias = model.layer1[0].bn2.bias
        model_truncation.layer1_0[0].bn2.running_mean = model.layer1[0].bn2.running_mean
        model_truncation.layer1_0[0].bn2.running_var = model.layer1[0].bn2.running_var

        model_truncation.layer1_1[0].conv1.weight = model.layer1[1].conv1.weight
        model_truncation.layer1_1[0].bn1.weight = model.layer1[1].bn1.weight
        model_truncation.layer1_1[0].bn1.bias = model.layer1[1].bn1.bias
        model_truncation.layer1_1[0].bn1.running_mean = model.layer1[1].bn1.running_mean
        model_truncation.layer1_1[0].bn1.running_var = model.layer1[1].bn1.running_var
        model_truncation.layer1_1[0].conv2.weight = model.layer1[1].conv2.weight
        model_truncation.layer1_1[0].bn2.weight = model.layer1[1].bn2.weight
        model_truncation.layer1_1[0].bn2.bias = model.layer1[1].bn2.bias
        model_truncation.layer1_1[0].bn2.running_mean = model.layer1[1].bn2.running_mean
        model_truncation.layer1_1[0].bn2.running_var = model.layer1[1].bn2.running_var

        model_truncation.layer2_0[0].conv1.weight = model.layer2[0].conv1.weight
        model_truncation.layer2_0[0].bn1.weight = model.layer2[0].bn1.weight
        model_truncation.layer2_0[0].bn1.bias = model.layer2[0].bn1.bias
        model_truncation.layer2_0[0].bn1.running_mean = model.layer2[0].bn1.running_mean
        model_truncation.layer2_0[0].bn1.running_var = model.layer2[0].bn1.running_var
        model_truncation.layer2_0[0].conv2.weight = model.layer2[0].conv2.weight
        model_truncation.layer2_0[0].bn2.weight = model.layer2[0].bn2.weight
        model_truncation.layer2_0[0].bn2.bias = model.layer2[0].bn2.bias
        model_truncation.layer2_0[0].bn2.running_mean = model.layer2[0].bn2.running_mean
        model_truncation.layer2_0[0].bn2.running_var = model.layer2[0].bn2.running_var
        model_truncation.layer2_0[0].downsample[0].weight = model.layer2[0].downsample[0].weight
        model_truncation.layer2_0[0].downsample[1].weight = model.layer2[0].downsample[1].weight
        model_truncation.layer2_0[0].downsample[1].bias = model.layer2[0].downsample[1].bias
        model_truncation.layer2_0[0].downsample[1].running_mean = model.layer2[0].downsample[1].running_mean
        model_truncation.layer2_0[0].downsample[1].running_var = model.layer2[0].downsample[1].running_var

        model_truncation.layer2_1[0].conv1.weight = model.layer2[1].conv1.weight
        model_truncation.layer2_1[0].bn1.weight = model.layer2[1].bn1.weight
        model_truncation.layer2_1[0].bn1.bias = model.layer2[1].bn1.bias
        model_truncation.layer2_1[0].bn1.running_mean = model.layer2[1].bn1.running_mean
        model_truncation.layer2_1[0].bn1.running_var = model.layer2[1].bn1.running_var
        model_truncation.layer2_1[0].conv2.weight = model.layer2[1].conv2.weight
        model_truncation.layer2_1[0].bn2.weight = model.layer2[1].bn2.weight
        model_truncation.layer2_1[0].bn2.bias = model.layer2[1].bn2.bias
        model_truncation.layer2_1[0].bn2.running_mean = model.layer2[1].bn2.running_mean
        model_truncation.layer2_1[0].bn2.running_var = model.layer2[1].bn2.running_var

        model_truncation.layer3_0[0].conv1.weight = model.layer3[0].conv1.weight
        model_truncation.layer3_0[0].bn1.weight = model.layer3[0].bn1.weight
        model_truncation.layer3_0[0].bn1.bias = model.layer3[0].bn1.bias
        model_truncation.layer3_0[0].bn1.running_mean = model.layer3[0].bn1.running_mean
        model_truncation.layer3_0[0].bn1.running_var = model.layer3[0].bn1.running_var
        model_truncation.layer3_0[0].conv2.weight = model.layer3[0].conv2.weight
        model_truncation.layer3_0[0].bn2.weight = model.layer3[0].bn2.weight
        model_truncation.layer3_0[0].bn2.bias = model.layer3[0].bn2.bias
        model_truncation.layer3_0[0].bn2.running_mean = model.layer3[0].bn2.running_mean
        model_truncation.layer3_0[0].bn2.running_var = model.layer3[0].bn2.running_var
        model_truncation.layer3_0[0].downsample[0].weight = model.layer3[0].downsample[0].weight
        model_truncation.layer3_0[0].downsample[1].weight = model.layer3[0].downsample[1].weight
        model_truncation.layer3_0[0].downsample[1].bias = model.layer3[0].downsample[1].bias
        model_truncation.layer3_0[0].downsample[1].running_mean = model.layer3[0].downsample[1].running_mean
        model_truncation.layer3_0[0].downsample[1].running_var = model.layer3[0].downsample[1].running_var

        model_truncation.layer3_1[0].conv1.weight = model.layer3[1].conv1.weight
        model_truncation.layer3_1[0].bn1.weight = model.layer3[1].bn1.weight
        model_truncation.layer3_1[0].bn1.bias = model.layer3[1].bn1.bias
        model_truncation.layer3_1[0].bn1.running_mean = model.layer3[1].bn1.running_mean
        model_truncation.layer3_1[0].bn1.running_var = model.layer3[1].bn1.running_var
        model_truncation.layer3_1[0].conv2.weight = model.layer3[1].conv2.weight
        model_truncation.layer3_1[0].bn2.weight = model.layer3[1].bn2.weight
        model_truncation.layer3_1[0].bn2.bias = model.layer3[1].bn2.bias
        model_truncation.layer3_1[0].bn2.running_mean = model.layer3[1].bn2.running_mean
        model_truncation.layer3_1[0].bn2.running_var = model.layer3[1].bn2.running_var

        model_truncation.layer4_0[0].conv1.weight = model.layer4[0].conv1.weight
        model_truncation.layer4_0[0].bn1.weight = model.layer4[0].bn1.weight
        model_truncation.layer4_0[0].bn1.bias = model.layer4[0].bn1.bias
        model_truncation.layer4_0[0].bn1.running_mean = model.layer4[0].bn1.running_mean
        model_truncation.layer4_0[0].bn1.running_var = model.layer4[0].bn1.running_var
        model_truncation.layer4_0[0].conv2.weight = model.layer4[0].conv2.weight
        model_truncation.layer4_0[0].bn2.weight = model.layer4[0].bn2.weight
        model_truncation.layer4_0[0].bn2.bias = model.layer4[0].bn2.bias
        model_truncation.layer4_0[0].bn2.running_mean = model.layer4[0].bn2.running_mean
        model_truncation.layer4_0[0].bn2.running_var = model.layer4[0].bn2.running_var
        model_truncation.layer4_0[0].downsample[0].weight = model.layer4[0].downsample[0].weight
        model_truncation.layer4_0[0].downsample[1].weight = model.layer4[0].downsample[1].weight
        model_truncation.layer4_0[0].downsample[1].bias = model.layer4[0].downsample[1].bias
        model_truncation.layer4_0[0].downsample[1].running_mean = model.layer4[0].downsample[1].running_mean
        model_truncation.layer4_0[0].downsample[1].running_var = model.layer4[0].downsample[1].running_var

        model_truncation.layer4_1[0].conv1.weight = model.layer4[1].conv1.weight
        model_truncation.layer4_1[0].bn1.weight = model.layer4[1].bn1.weight
        model_truncation.layer4_1[0].bn1.bias = model.layer4[1].bn1.bias
        model_truncation.layer4_1[0].bn1.running_mean = model.layer4[1].bn1.running_mean
        model_truncation.layer4_1[0].bn1.running_var = model.layer4[1].bn1.running_var
        model_truncation.layer4_1[0].conv2.weight = model.layer4[1].conv2.weight
        model_truncation.layer4_1[0].bn2.weight = model.layer4[1].bn2.weight
        model_truncation.layer4_1[0].bn2.bias = model.layer4[1].bn2.bias
        model_truncation.layer4_1[0].bn2.running_mean = model.layer4[1].bn2.running_mean
        model_truncation.layer4_1[0].bn2.running_var = model.layer4[1].bn2.running_var

        model_truncation.fc.linear.weight = model.fc.linear.weight
        model_truncation.fc.linear.bias = model.fc.linear.bias
    
    elif resnet_model == "Resnet50":
        model_truncation.conv1.weight = model.conv1.weight
        model_truncation.bn1.weight = model.bn1.weight
        model_truncation.bn1.bias = model.bn1.bias
        model_truncation.bn1.running_mean = model.bn1.running_mean
        model_truncation.bn1.running_var = model.bn1.running_var

        model_truncation.layer1_0[0].conv1.weight = model.layer1[0].conv1.weight
        model_truncation.layer1_0[0].bn1.weight = model.layer1[0].bn1.weight
        model_truncation.layer1_0[0].bn1.bias = model.layer1[0].bn1.bias
        model_truncation.layer1_0[0].bn1.running_mean = model.layer1[0].bn1.running_mean
        model_truncation.layer1_0[0].bn1.running_var = model.layer1[0].bn1.running_var
        model_truncation.layer1_0[0].conv2.weight = model.layer1[0].conv2.weight
        model_truncation.layer1_0[0].bn2.weight = model.layer1[0].bn2.weight
        model_truncation.layer1_0[0].bn2.bias = model.layer1[0].bn2.bias
        model_truncation.layer1_0[0].bn2.running_mean = model.layer1[0].bn2.running_mean
        model_truncation.layer1_0[0].bn2.running_var = model.layer1[0].bn2.running_var
        model_truncation.layer1_0[0].conv3.weight = model.layer1[0].conv3.weight
        model_truncation.layer1_0[0].bn3.weight = model.layer1[0].bn3.weight
        model_truncation.layer1_0[0].bn3.bias = model.layer1[0].bn3.bias
        model_truncation.layer1_0[0].bn3.running_mean = model.layer1[0].bn3.running_mean
        model_truncation.layer1_0[0].bn3.running_var = model.layer1[0].bn3.running_var
        model_truncation.layer1_0[0].downsample[0].weight = model.layer1[0].downsample[0].weight
        model_truncation.layer1_0[0].downsample[1].weight = model.layer1[0].downsample[1].weight
        model_truncation.layer1_0[0].downsample[1].bias = model.layer1[0].downsample[1].bias
        model_truncation.layer1_0[0].downsample[1].running_mean = model.layer1[0].downsample[1].running_mean
        model_truncation.layer1_0[0].downsample[1].running_var = model.layer1[0].downsample[1].running_var

        model_truncation.layer1_1[0].conv1.weight = model.layer1[1].conv1.weight
        model_truncation.layer1_1[0].bn1.weight = model.layer1[1].bn1.weight
        model_truncation.layer1_1[0].bn1.bias = model.layer1[1].bn1.bias
        model_truncation.layer1_1[0].bn1.running_mean = model.layer1[1].bn1.running_mean
        model_truncation.layer1_1[0].bn1.running_var = model.layer1[1].bn1.running_var
        model_truncation.layer1_1[0].conv2.weight = model.layer1[1].conv2.weight
        model_truncation.layer1_1[0].bn2.weight = model.layer1[1].bn2.weight
        model_truncation.layer1_1[0].bn2.bias = model.layer1[1].bn2.bias
        model_truncation.layer1_1[0].bn2.running_mean = model.layer1[1].bn2.running_mean
        model_truncation.layer1_1[0].bn2.running_var = model.layer1[1].bn2.running_var
        model_truncation.layer1_1[0].conv3.weight = model.layer1[1].conv3.weight
        model_truncation.layer1_1[0].bn3.weight = model.layer1[1].bn3.weight
        model_truncation.layer1_1[0].bn3.bias = model.layer1[1].bn3.bias
        model_truncation.layer1_1[0].bn3.running_mean = model.layer1[1].bn3.running_mean
        model_truncation.layer1_1[0].bn3.running_var = model.layer1[1].bn3.running_var

        model_truncation.layer1_2[0].conv1.weight = model.layer1[2].conv1.weight
        model_truncation.layer1_2[0].bn1.weight = model.layer1[2].bn1.weight
        model_truncation.layer1_2[0].bn1.bias = model.layer1[2].bn1.bias
        model_truncation.layer1_2[0].bn1.running_mean = model.layer1[2].bn1.running_mean
        model_truncation.layer1_2[0].bn1.running_var = model.layer1[2].bn1.running_var
        model_truncation.layer1_2[0].conv2.weight = model.layer1[2].conv2.weight
        model_truncation.layer1_2[0].bn2.weight = model.layer1[2].bn2.weight
        model_truncation.layer1_2[0].bn2.bias = model.layer1[2].bn2.bias
        model_truncation.layer1_2[0].bn2.running_mean = model.layer1[2].bn2.running_mean
        model_truncation.layer1_2[0].bn2.running_var = model.layer1[2].bn2.running_var
        model_truncation.layer1_2[0].conv3.weight = model.layer1[2].conv3.weight
        model_truncation.layer1_2[0].bn3.weight = model.layer1[2].bn3.weight
        model_truncation.layer1_2[0].bn3.bias = model.layer1[2].bn3.bias
        model_truncation.layer1_2[0].bn3.running_mean = model.layer1[2].bn3.running_mean
        model_truncation.layer1_2[0].bn3.running_var = model.layer1[2].bn3.running_var

        model_truncation.layer2_0[0].conv1.weight = model.layer2[0].conv1.weight
        model_truncation.layer2_0[0].bn1.weight = model.layer2[0].bn1.weight
        model_truncation.layer2_0[0].bn1.bias = model.layer2[0].bn1.bias
        model_truncation.layer2_0[0].bn1.running_mean = model.layer2[0].bn1.running_mean
        model_truncation.layer2_0[0].bn1.running_var = model.layer2[0].bn1.running_var
        model_truncation.layer2_0[0].conv2.weight = model.layer2[0].conv2.weight
        model_truncation.layer2_0[0].bn2.weight = model.layer2[0].bn2.weight
        model_truncation.layer2_0[0].bn2.bias = model.layer2[0].bn2.bias
        model_truncation.layer2_0[0].bn2.running_mean = model.layer2[0].bn2.running_mean
        model_truncation.layer2_0[0].bn2.running_var = model.layer2[0].bn2.running_var
        model_truncation.layer2_0[0].conv3.weight = model.layer2[0].conv3.weight
        model_truncation.layer2_0[0].bn3.weight = model.layer2[0].bn3.weight
        model_truncation.layer2_0[0].bn3.bias = model.layer2[0].bn3.bias
        model_truncation.layer2_0[0].bn3.running_mean = model.layer2[0].bn3.running_mean
        model_truncation.layer2_0[0].bn3.running_var = model.layer2[0].bn3.running_var
        model_truncation.layer2_0[0].downsample[0].weight = model.layer2[0].downsample[0].weight
        model_truncation.layer2_0[0].downsample[1].weight = model.layer2[0].downsample[1].weight
        model_truncation.layer2_0[0].downsample[1].bias = model.layer2[0].downsample[1].bias
        model_truncation.layer2_0[0].downsample[1].running_mean = model.layer2[0].downsample[1].running_mean
        model_truncation.layer2_0[0].downsample[1].running_var = model.layer2[0].downsample[1].running_var

        model_truncation.layer2_1[0].conv1.weight = model.layer2[1].conv1.weight
        model_truncation.layer2_1[0].bn1.weight = model.layer2[1].bn1.weight
        model_truncation.layer2_1[0].bn1.bias = model.layer2[1].bn1.bias
        model_truncation.layer2_1[0].bn1.running_mean = model.layer2[1].bn1.running_mean
        model_truncation.layer2_1[0].bn1.running_var = model.layer2[1].bn1.running_var
        model_truncation.layer2_1[0].conv2.weight = model.layer2[1].conv2.weight
        model_truncation.layer2_1[0].bn2.weight = model.layer2[1].bn2.weight
        model_truncation.layer2_1[0].bn2.bias = model.layer2[1].bn2.bias
        model_truncation.layer2_1[0].bn2.running_mean = model.layer2[1].bn2.running_mean
        model_truncation.layer2_1[0].bn2.running_var = model.layer2[1].bn2.running_var
        model_truncation.layer2_1[0].conv3.weight = model.layer2[1].conv3.weight
        model_truncation.layer2_1[0].bn3.weight = model.layer2[1].bn3.weight
        model_truncation.layer2_1[0].bn3.bias = model.layer2[1].bn3.bias
        model_truncation.layer2_1[0].bn3.running_mean = model.layer2[1].bn3.running_mean
        model_truncation.layer2_1[0].bn3.running_var = model.layer2[1].bn3.running_var

        model_truncation.layer2_2[0].conv1.weight = model.layer2[2].conv1.weight
        model_truncation.layer2_2[0].bn1.weight = model.layer2[2].bn1.weight
        model_truncation.layer2_2[0].bn1.bias = model.layer2[2].bn1.bias
        model_truncation.layer2_2[0].bn1.running_mean = model.layer2[2].bn1.running_mean
        model_truncation.layer2_2[0].bn1.running_var = model.layer2[2].bn1.running_var
        model_truncation.layer2_2[0].conv2.weight = model.layer2[2].conv2.weight
        model_truncation.layer2_2[0].bn2.weight = model.layer2[2].bn2.weight
        model_truncation.layer2_2[0].bn2.bias = model.layer2[2].bn2.bias
        model_truncation.layer2_2[0].bn2.running_mean = model.layer2[2].bn2.running_mean
        model_truncation.layer2_2[0].bn2.running_var = model.layer2[2].bn2.running_var
        model_truncation.layer2_2[0].conv3.weight = model.layer2[2].conv3.weight
        model_truncation.layer2_2[0].bn3.weight = model.layer2[2].bn3.weight
        model_truncation.layer2_2[0].bn3.bias = model.layer2[2].bn3.bias
        model_truncation.layer2_2[0].bn3.running_mean = model.layer2[2].bn3.running_mean
        model_truncation.layer2_2[0].bn3.running_var = model.layer2[2].bn3.running_var

        model_truncation.layer2_3[0].conv1.weight = model.layer2[3].conv1.weight
        model_truncation.layer2_3[0].bn1.weight = model.layer2[3].bn1.weight
        model_truncation.layer2_3[0].bn1.bias = model.layer2[3].bn1.bias
        model_truncation.layer2_3[0].bn1.running_mean = model.layer2[3].bn1.running_mean
        model_truncation.layer2_3[0].bn1.running_var = model.layer2[3].bn1.running_var
        model_truncation.layer2_3[0].conv2.weight = model.layer2[3].conv2.weight
        model_truncation.layer2_3[0].bn2.weight = model.layer2[3].bn2.weight
        model_truncation.layer2_3[0].bn2.bias = model.layer2[3].bn2.bias
        model_truncation.layer2_3[0].bn2.running_mean = model.layer2[3].bn2.running_mean
        model_truncation.layer2_3[0].bn2.running_var = model.layer2[3].bn2.running_var
        model_truncation.layer2_3[0].conv3.weight = model.layer2[3].conv3.weight
        model_truncation.layer2_3[0].bn3.weight = model.layer2[3].bn3.weight
        model_truncation.layer2_3[0].bn3.bias = model.layer2[3].bn3.bias
        model_truncation.layer2_3[0].bn3.running_mean = model.layer2[3].bn3.running_mean
        model_truncation.layer2_3[0].bn3.running_var = model.layer2[3].bn3.running_var

        model_truncation.layer3_0[0].conv1.weight = model.layer3[0].conv1.weight
        model_truncation.layer3_0[0].bn1.weight = model.layer3[0].bn1.weight
        model_truncation.layer3_0[0].bn1.bias = model.layer3[0].bn1.bias
        model_truncation.layer3_0[0].bn1.running_mean = model.layer3[0].bn1.running_mean
        model_truncation.layer3_0[0].bn1.running_var = model.layer3[0].bn1.running_var
        model_truncation.layer3_0[0].conv2.weight = model.layer3[0].conv2.weight
        model_truncation.layer3_0[0].bn2.weight = model.layer3[0].bn2.weight
        model_truncation.layer3_0[0].bn2.bias = model.layer3[0].bn2.bias
        model_truncation.layer3_0[0].bn2.running_mean = model.layer3[0].bn2.running_mean
        model_truncation.layer3_0[0].bn2.running_var = model.layer3[0].bn2.running_var
        model_truncation.layer3_0[0].conv3.weight = model.layer3[0].conv3.weight
        model_truncation.layer3_0[0].bn3.weight = model.layer3[0].bn3.weight
        model_truncation.layer3_0[0].bn3.bias = model.layer3[0].bn3.bias
        model_truncation.layer3_0[0].bn3.running_mean = model.layer3[0].bn3.running_mean
        model_truncation.layer3_0[0].bn3.running_var = model.layer3[0].bn3.running_var
        model_truncation.layer3_0[0].downsample[0].weight = model.layer3[0].downsample[0].weight
        model_truncation.layer3_0[0].downsample[1].weight = model.layer3[0].downsample[1].weight
        model_truncation.layer3_0[0].downsample[1].bias = model.layer3[0].downsample[1].bias
        model_truncation.layer3_0[0].downsample[1].running_mean = model.layer3[0].downsample[1].running_mean
        model_truncation.layer3_0[0].downsample[1].running_var = model.layer3[0].downsample[1].running_var

        model_truncation.layer3_1[0].conv1.weight = model.layer3[1].conv1.weight
        model_truncation.layer3_1[0].bn1.weight = model.layer3[1].bn1.weight
        model_truncation.layer3_1[0].bn1.bias = model.layer3[1].bn1.bias
        model_truncation.layer3_1[0].bn1.running_mean = model.layer3[1].bn1.running_mean
        model_truncation.layer3_1[0].bn1.running_var = model.layer3[1].bn1.running_var
        model_truncation.layer3_1[0].conv2.weight = model.layer3[1].conv2.weight
        model_truncation.layer3_1[0].bn2.weight = model.layer3[1].bn2.weight
        model_truncation.layer3_1[0].bn2.bias = model.layer3[1].bn2.bias
        model_truncation.layer3_1[0].bn2.running_mean = model.layer3[1].bn2.running_mean
        model_truncation.layer3_1[0].bn2.running_var = model.layer3[1].bn2.running_var
        model_truncation.layer3_1[0].conv3.weight = model.layer3[1].conv3.weight
        model_truncation.layer3_1[0].bn3.weight = model.layer3[1].bn3.weight
        model_truncation.layer3_1[0].bn3.bias = model.layer3[1].bn3.bias
        model_truncation.layer3_1[0].bn3.running_mean = model.layer3[1].bn3.running_mean
        model_truncation.layer3_1[0].bn3.running_var = model.layer3[1].bn3.running_var

        model_truncation.layer3_2[0].conv1.weight = model.layer3[2].conv1.weight
        model_truncation.layer3_2[0].bn1.weight = model.layer3[2].bn1.weight
        model_truncation.layer3_2[0].bn1.bias = model.layer3[2].bn1.bias
        model_truncation.layer3_2[0].bn1.running_mean = model.layer3[2].bn1.running_mean
        model_truncation.layer3_2[0].bn1.running_var = model.layer3[2].bn1.running_var
        model_truncation.layer3_2[0].conv2.weight = model.layer3[2].conv2.weight
        model_truncation.layer3_2[0].bn2.weight = model.layer3[2].bn2.weight
        model_truncation.layer3_2[0].bn2.bias = model.layer3[2].bn2.bias
        model_truncation.layer3_2[0].bn2.running_mean = model.layer3[2].bn2.running_mean
        model_truncation.layer3_2[0].bn2.running_var = model.layer3[2].bn2.running_var
        model_truncation.layer3_2[0].conv3.weight = model.layer3[2].conv3.weight
        model_truncation.layer3_2[0].bn3.weight = model.layer3[2].bn3.weight
        model_truncation.layer3_2[0].bn3.bias = model.layer3[2].bn3.bias
        model_truncation.layer3_2[0].bn3.running_mean = model.layer3[2].bn3.running_mean
        model_truncation.layer3_2[0].bn3.running_var = model.layer3[2].bn3.running_var

        model_truncation.layer3_3[0].conv1.weight = model.layer3[3].conv1.weight
        model_truncation.layer3_3[0].bn1.weight = model.layer3[3].bn1.weight
        model_truncation.layer3_3[0].bn1.bias = model.layer3[3].bn1.bias
        model_truncation.layer3_3[0].bn1.running_mean = model.layer3[3].bn1.running_mean
        model_truncation.layer3_3[0].bn1.running_var = model.layer3[3].bn1.running_var
        model_truncation.layer3_3[0].conv2.weight = model.layer3[3].conv2.weight
        model_truncation.layer3_3[0].bn2.weight = model.layer3[3].bn2.weight
        model_truncation.layer3_3[0].bn2.bias = model.layer3[3].bn2.bias
        model_truncation.layer3_3[0].bn2.running_mean = model.layer3[3].bn2.running_mean
        model_truncation.layer3_3[0].bn2.running_var = model.layer3[3].bn2.running_var
        model_truncation.layer3_3[0].conv3.weight = model.layer3[3].conv3.weight
        model_truncation.layer3_3[0].bn3.weight = model.layer3[3].bn3.weight
        model_truncation.layer3_3[0].bn3.bias = model.layer3[3].bn3.bias
        model_truncation.layer3_3[0].bn3.running_mean = model.layer3[3].bn3.running_mean
        model_truncation.layer3_3[0].bn3.running_var = model.layer3[3].bn3.running_var

        model_truncation.layer3_4[0].conv1.weight = model.layer3[4].conv1.weight
        model_truncation.layer3_4[0].bn1.weight = model.layer3[4].bn1.weight
        model_truncation.layer3_4[0].bn1.bias = model.layer3[4].bn1.bias
        model_truncation.layer3_4[0].bn1.running_mean = model.layer3[4].bn1.running_mean
        model_truncation.layer3_4[0].bn1.running_var = model.layer3[4].bn1.running_var
        model_truncation.layer3_4[0].conv2.weight = model.layer3[4].conv2.weight
        model_truncation.layer3_4[0].bn2.weight = model.layer3[4].bn2.weight
        model_truncation.layer3_4[0].bn2.bias = model.layer3[4].bn2.bias
        model_truncation.layer3_4[0].bn2.running_mean = model.layer3[4].bn2.running_mean
        model_truncation.layer3_4[0].bn2.running_var = model.layer3[4].bn2.running_var
        model_truncation.layer3_4[0].conv3.weight = model.layer3[4].conv3.weight
        model_truncation.layer3_4[0].bn3.weight = model.layer3[4].bn3.weight
        model_truncation.layer3_4[0].bn3.bias = model.layer3[4].bn3.bias
        model_truncation.layer3_4[0].bn3.running_mean = model.layer3[4].bn3.running_mean
        model_truncation.layer3_4[0].bn3.running_var = model.layer3[4].bn3.running_var

        model_truncation.layer3_5[0].conv1.weight = model.layer3[5].conv1.weight
        model_truncation.layer3_5[0].bn1.weight = model.layer3[5].bn1.weight
        model_truncation.layer3_5[0].bn1.bias = model.layer3[5].bn1.bias
        model_truncation.layer3_5[0].bn1.running_mean = model.layer3[5].bn1.running_mean
        model_truncation.layer3_5[0].bn1.running_var = model.layer3[5].bn1.running_var
        model_truncation.layer3_5[0].conv2.weight = model.layer3[5].conv2.weight
        model_truncation.layer3_5[0].bn2.weight = model.layer3[5].bn2.weight
        model_truncation.layer3_5[0].bn2.bias = model.layer3[5].bn2.bias
        model_truncation.layer3_5[0].bn2.running_mean = model.layer3[5].bn2.running_mean
        model_truncation.layer3_5[0].bn2.running_var = model.layer3[5].bn2.running_var
        model_truncation.layer3_5[0].conv3.weight = model.layer3[5].conv3.weight
        model_truncation.layer3_5[0].bn3.weight = model.layer3[5].bn3.weight
        model_truncation.layer3_5[0].bn3.bias = model.layer3[5].bn3.bias
        model_truncation.layer3_5[0].bn3.running_mean = model.layer3[5].bn3.running_mean
        model_truncation.layer3_5[0].bn3.running_var = model.layer3[5].bn3.running_var

        model_truncation.layer4_0[0].conv1.weight = model.layer4[0].conv1.weight
        model_truncation.layer4_0[0].bn1.weight = model.layer4[0].bn1.weight
        model_truncation.layer4_0[0].bn1.bias = model.layer4[0].bn1.bias
        model_truncation.layer4_0[0].bn1.running_mean = model.layer4[0].bn1.running_mean
        model_truncation.layer4_0[0].bn1.running_var = model.layer4[0].bn1.running_var
        model_truncation.layer4_0[0].conv2.weight = model.layer4[0].conv2.weight
        model_truncation.layer4_0[0].bn2.weight = model.layer4[0].bn2.weight
        model_truncation.layer4_0[0].bn2.bias = model.layer4[0].bn2.bias
        model_truncation.layer4_0[0].bn2.running_mean = model.layer4[0].bn2.running_mean
        model_truncation.layer4_0[0].bn2.running_var = model.layer4[0].bn2.running_var
        model_truncation.layer4_0[0].conv3.weight = model.layer4[0].conv3.weight
        model_truncation.layer4_0[0].bn3.weight = model.layer4[0].bn3.weight
        model_truncation.layer4_0[0].bn3.bias = model.layer4[0].bn3.bias
        model_truncation.layer4_0[0].bn3.running_mean = model.layer4[0].bn3.running_mean
        model_truncation.layer4_0[0].bn3.running_var = model.layer4[0].bn3.running_var
        model_truncation.layer4_0[0].downsample[0].weight = model.layer4[0].downsample[0].weight
        model_truncation.layer4_0[0].downsample[1].weight = model.layer4[0].downsample[1].weight
        model_truncation.layer4_0[0].downsample[1].bias = model.layer4[0].downsample[1].bias
        model_truncation.layer4_0[0].downsample[1].running_mean = model.layer4[0].downsample[1].running_mean
        model_truncation.layer4_0[0].downsample[1].running_var = model.layer4[0].downsample[1].running_var

        model_truncation.layer4_1[0].conv1.weight = model.layer4[1].conv1.weight
        model_truncation.layer4_1[0].bn1.weight = model.layer4[1].bn1.weight
        model_truncation.layer4_1[0].bn1.bias = model.layer4[1].bn1.bias
        model_truncation.layer4_1[0].bn1.running_mean = model.layer4[1].bn1.running_mean
        model_truncation.layer4_1[0].bn1.running_var = model.layer4[1].bn1.running_var
        model_truncation.layer4_1[0].conv2.weight = model.layer4[1].conv2.weight
        model_truncation.layer4_1[0].bn2.weight = model.layer4[1].bn2.weight
        model_truncation.layer4_1[0].bn2.bias = model.layer4[1].bn2.bias
        model_truncation.layer4_1[0].bn2.running_mean = model.layer4[1].bn2.running_mean
        model_truncation.layer4_1[0].bn2.running_var = model.layer4[1].bn2.running_var
        model_truncation.layer4_1[0].conv3.weight = model.layer4[1].conv3.weight
        model_truncation.layer4_1[0].bn3.weight = model.layer4[1].bn3.weight
        model_truncation.layer4_1[0].bn3.bias = model.layer4[1].bn3.bias
        model_truncation.layer4_1[0].bn3.running_mean = model.layer4[1].bn3.running_mean
        model_truncation.layer4_1[0].bn3.running_var = model.layer4[1].bn3.running_var

        model_truncation.layer4_2[0].conv1.weight = model.layer4[2].conv1.weight
        model_truncation.layer4_2[0].bn1.weight = model.layer4[2].bn1.weight
        model_truncation.layer4_2[0].bn1.bias = model.layer4[2].bn1.bias
        model_truncation.layer4_2[0].bn1.running_mean = model.layer4[2].bn1.running_mean
        model_truncation.layer4_2[0].bn1.running_var = model.layer4[2].bn1.running_var
        model_truncation.layer4_2[0].conv2.weight = model.layer4[2].conv2.weight
        model_truncation.layer4_2[0].bn2.weight = model.layer4[2].bn2.weight
        model_truncation.layer4_2[0].bn2.bias = model.layer4[2].bn2.bias
        model_truncation.layer4_2[0].bn2.running_mean = model.layer4[2].bn2.running_mean
        model_truncation.layer4_2[0].bn2.running_var = model.layer4[2].bn2.running_var
        model_truncation.layer4_2[0].conv3.weight = model.layer4[2].conv3.weight
        model_truncation.layer4_2[0].bn3.weight = model.layer4[2].bn3.weight
        model_truncation.layer4_2[0].bn3.bias = model.layer4[2].bn3.bias
        model_truncation.layer4_2[0].bn3.running_mean = model.layer4[2].bn3.running_mean
        model_truncation.layer4_2[0].bn3.running_var = model.layer4[2].bn3.running_var

        model_truncation.fc.linear.weight = model.fc.linear.weight
        model_truncation.fc.linear.bias = model.fc.linear.bias

    elif resnet_model == "VGG16":
        model_truncation.conv1.weight = model.features[0].weight
        model_truncation.conv1.bias = model.features[0].bias
        model_truncation.bn1.weight = model.features[1].weight
        model_truncation.bn1.bias = model.features[1].bias
        model_truncation.bn1.running_mean = model.features[1].running_mean
        model_truncation.bn1.running_var = model.features[1].running_var

        model_truncation.conv2.weight = model.features[3].weight
        model_truncation.conv2.bias = model.features[3].bias
        model_truncation.bn2.weight = model.features[4].weight
        model_truncation.bn2.bias = model.features[4].bias
        model_truncation.bn2.running_mean = model.features[4].running_mean
        model_truncation.bn2.running_var = model.features[4].running_var
        
        model_truncation.conv3.weight = model.features[7].weight
        model_truncation.conv3.bias = model.features[7].bias
        model_truncation.bn3.weight = model.features[8].weight
        model_truncation.bn3.bias = model.features[8].bias
        model_truncation.bn3.running_mean = model.features[8].running_mean
        model_truncation.bn3.running_var = model.features[8].running_var

        model_truncation.conv4.weight = model.features[10].weight
        model_truncation.conv4.bias = model.features[10].bias
        model_truncation.bn4.weight = model.features[11].weight
        model_truncation.bn4.bias = model.features[11].bias
        model_truncation.bn4.running_mean = model.features[11].running_mean
        model_truncation.bn4.running_var = model.features[11].running_var

        model_truncation.conv5.weight = model.features[14].weight
        model_truncation.conv5.bias = model.features[14].bias
        model_truncation.bn5.weight = model.features[15].weight
        model_truncation.bn5.bias = model.features[15].bias
        model_truncation.bn5.running_mean = model.features[15].running_mean
        model_truncation.bn5.running_var = model.features[15].running_var

        model_truncation.conv6.weight = model.features[17].weight
        model_truncation.conv6.bias = model.features[17].bias
        model_truncation.bn6.weight = model.features[18].weight
        model_truncation.bn6.bias = model.features[18].bias
        model_truncation.bn6.running_mean = model.features[18].running_mean
        model_truncation.bn6.running_var = model.features[18].running_var

        model_truncation.conv7.weight = model.features[20].weight
        model_truncation.conv7.bias = model.features[20].bias
        model_truncation.bn7.weight = model.features[21].weight
        model_truncation.bn7.bias = model.features[21].bias
        model_truncation.bn7.running_mean = model.features[21].running_mean
        model_truncation.bn7.running_var = model.features[21].running_var

        model_truncation.conv8.weight = model.features[24].weight
        model_truncation.conv8.bias = model.features[24].bias
        model_truncation.bn8.weight = model.features[25].weight
        model_truncation.bn8.bias = model.features[25].bias
        model_truncation.bn8.running_mean = model.features[25].running_mean
        model_truncation.bn8.running_var = model.features[25].running_var

        model_truncation.conv9.weight = model.features[27].weight
        model_truncation.conv9.bias = model.features[27].bias
        model_truncation.bn9.weight = model.features[28].weight
        model_truncation.bn9.bias = model.features[28].bias
        model_truncation.bn9.running_mean = model.features[28].running_mean
        model_truncation.bn9.running_var = model.features[28].running_var

        model_truncation.conv10.weight = model.features[30].weight
        model_truncation.conv10.bias = model.features[30].bias
        model_truncation.bn10.weight = model.features[31].weight
        model_truncation.bn10.bias = model.features[31].bias
        model_truncation.bn10.running_mean = model.features[31].running_mean
        model_truncation.bn10.running_var = model.features[31].running_var

        model_truncation.conv11.weight = model.features[34].weight
        model_truncation.conv11.bias = model.features[34].bias
        model_truncation.bn11.weight = model.features[35].weight
        model_truncation.bn11.bias = model.features[35].bias
        model_truncation.bn11.running_mean = model.features[35].running_mean
        model_truncation.bn11.running_var = model.features[35].running_var

        model_truncation.conv12.weight = model.features[37].weight
        model_truncation.conv12.bias = model.features[37].bias
        model_truncation.bn12.weight = model.features[38].weight
        model_truncation.bn12.bias = model.features[38].bias
        model_truncation.bn12.running_mean = model.features[38].running_mean
        model_truncation.bn12.running_var = model.features[38].running_var

        model_truncation.conv13.weight = model.features[40].weight
        model_truncation.conv13.bias = model.features[40].bias
        model_truncation.bn13.weight = model.features[41].weight
        model_truncation.bn13.bias = model.features[41].bias
        model_truncation.bn13.running_mean = model.features[41].running_mean
        model_truncation.bn13.running_var = model.features[41].running_var

        model_truncation.classifier[0].weight = model.classifier[0].weight
        model_truncation.classifier[0].bias = model.classifier[0].bias
        model_truncation.classifier[3].weight = model.classifier[3].weight
        model_truncation.classifier[3].bias = model.classifier[3].bias
        model_truncation.classifier[6].linear.weight = model.classifier[6].linear.weight
        model_truncation.classifier[6].linear.bias = model.classifier[6].linear.bias

    return model_truncation