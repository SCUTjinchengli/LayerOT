import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .vgg import vgg16_bn
from .vgg16_truncation import vgg16_trunc
from .resnet import resnet18, resnet50
from .resnet18_truncation import resnet18_trunc
from .resnet50_truncation import resnet50_trunc
from .initial_truncation_model import get_additional_layer, initialization_model_truncation


class SigmoidLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SigmoidLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        linear_params = list(self.linear.parameters())
        linear_params[0].data.normal_(0, 0.01)
        linear_params[0].data.fill_(0)
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    

def load_model(opt, num_classes):
    # load templet
    if opt.model == "Alexnet":
        templet = AlexnetTemplet(opt.input_channel, opt.pretrain)
    elif opt.model == "Resnet18":
        if opt.truncation != "":
            # load the original model
            templet = resnet18(False, True, None)
            num_features = templet.fc.in_features
            templet.fc = SigmoidLinear(num_features, num_classes)
            # load the truncation model
            model_truncation = resnet18_trunc(False, True, opt.truncation)
            num_features = model_truncation.fc.in_features
            model_truncation.fc = SigmoidLinear(num_features, num_classes)
            logging.info(model_truncation)
        else:
            templet = resnet18(opt.pretrain, True, None)
            num_features = templet.fc.in_features
            templet.fc = SigmoidLinear(num_features, num_classes)
    elif opt.model == "Resnet50":
        if opt.truncation != "":
            # load the original model
            templet = resnet50(False, True, None)
            num_features = templet.fc.in_features
            templet.fc = SigmoidLinear(num_features, num_classes)
            # load the truncation model
            model_truncation = resnet50_trunc(False, True, opt.truncation)
            num_features = model_truncation.fc.in_features
            model_truncation.fc = SigmoidLinear(num_features, num_classes)
            logging.info(model_truncation)
        else:        
            templet = resnet50(opt.pretrain, True, None)
            num_features = templet.fc.in_features
            templet.fc = SigmoidLinear(num_features, num_classes)
    elif opt.model == "VGG16":
        if opt.truncation != "":
            # load the original model
            templet = vgg16_bn(False, True)
            num_features = templet.classifier[6].in_features
            templet.classifier[6] = SigmoidLinear(num_features, num_classes)
            # load the truncation model
            model_truncation = vgg16_trunc(False, True, opt.truncation)
            num_features = model_truncation.classifier[6].in_features
            model_truncation.classifier[6] = SigmoidLinear(num_features, num_classes)
            logging.info(model_truncation)
        else:        
            templet = vgg16_bn(opt.pretrain, True)
            num_features = templet.classifier[6].in_features
            templet.classifier[6] = SigmoidLinear(num_features, num_classes)
    else:
        logging.error("unknown model type")
        sys.exit(0)
    
    model = templet
    logging.info(model)

    # imagenet pretrain model
    if opt.pretrain:
        logging.info("use imagenet pretrained model")
    
    # load exsiting model
    if opt.checkpoint_name != "":
        if os.path.exists(opt.checkpoint_name):
            logging.info("load pretrained model from "+opt.checkpoint_name)
            model.load_state_dict(torch.load(opt.checkpoint_name))
        elif os.path.exists(opt.model_dir):
            checkpoint_name = opt.model_dir + "/" + opt.checkpoint_name
            model.load_state_dict(torch.load(checkpoint_name))
            logging.info("load pretrained model from "+ checkpoint_name)
        else:
            assert False
    
    # model_truncation parameters initialization by exsiting model
    if opt.truncation != "":    
        checkpoint_name = opt.dir + "/trainer_model/Train/epoch_100_snapshot.pth"
        model.load_state_dict(torch.load(checkpoint_name))
        logging.info("load pretrained model from "+ checkpoint_name)
        model_truncation = initialization_model_truncation(opt.model, model_truncation, model)
    
    if opt.truncation != "":
        templet2 = get_additional_layer(opt.model, opt.truncation, num_classes)
        model_addition = templet2
        logging.info(model_addition)
        
        if opt.truncation_checkpoint_name != "":
            if os.path.exists(opt.truncation_checkpoint_name):
                logging.info("load pretrained model_addition from "+opt.truncation_checkpoint_name)
                model_addition.load_state_dict(torch.load(opt.truncation_checkpoint_name))
            elif os.path.exists(opt.model_dir):
                truncation_checkpoint_name = opt.model_dir + "/" + opt.truncation_checkpoint_name
                model_addition.load_state_dict(torch.load(truncation_checkpoint_name))
                logging.info("load pretrained model from "+ truncation_checkpoint_name)
            else:
                assert False
        return model, model_truncation, model_addition
    else:
        return model


def save_model(model, opt, epoch, best):
    if best:
        checkpoint_name = opt.model_dir + "/best_model_snapshot.pth"
        torch.save(model.cpu().state_dict(), checkpoint_name)
        if opt.cuda and torch.cuda.is_available():
            model.cuda(opt.devices[0])
    else:
        checkpoint_name = opt.model_dir + "/epoch_%s_snapshot.pth" %(epoch)
        torch.save(model.cpu().state_dict(), checkpoint_name)
        if opt.cuda and torch.cuda.is_available():
            model.cuda(opt.devices[0])

def save_model_addition_layer(model, AddtionalLayer0, AddtionalLayer1, AddtionalLayer2, AddtionalLayer3, AddtionalLayer4, opt, epoch, best):

    checkpoint_name = opt.model_dir + "/epoch_%s_snapshot.pth" %(epoch)
    state = {'model':model.cpu().state_dict(),
            'AddtionalLayer0':AddtionalLayer0.cpu().state_dict(),
            'AddtionalLayer1':AddtionalLayer1.cpu().state_dict(),
            'AddtionalLayer2':AddtionalLayer2.cpu().state_dict(),
            'AddtionalLayer3':AddtionalLayer3.cpu().state_dict(),
            'AddtionalLayer4':AddtionalLayer4.cpu().state_dict()
            }

    torch.save(state, checkpoint_name)

    if opt.cuda and torch.cuda.is_available():
        model.cuda(opt.devices[0])
        AddtionalLayer0.cuda(opt.devices[0])
        AddtionalLayer1.cuda(opt.devices[0])
        AddtionalLayer2.cuda(opt.devices[0])
        AddtionalLayer3.cuda(opt.devices[0])
        AddtionalLayer4.cuda(opt.devices[0])
        


def modify_last_layer_lr(named_params, base_lr, lr_mult_w, lr_mult_b):
    params = list()
    for name, param in named_params: 
        if 'bias' in name:
            if 'FullyConnectedLayer_' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_b, 'weight_decay': 0}]
            else:
                params += [{'params':param, 'lr': base_lr * 2, 'weight_decay': 0}]
        else:
            if 'FullyConnectedLayer_' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_w}]
            else:
                params += [{'params':param, 'lr': base_lr * 1}]
    return params