import os
import torch.nn as nn

from options.options import Options
from models.model import load_model
from trainer import *


def main_truncation(opt):
    # prepare
    initialize(opt)
    train_set, test_set = get_dataset(opt)

    C = get_distance_matrix(opt.dataset_name)
    criterion_BCE = nn.BCELoss()
    criterion_KL = nn.KLDivLoss(reduction='batchmean')
    # the model has loaded the exsiting model parameters
    model, model_truncation, AdditionalLayer = load_model(opt, opt.num_classes)    
    if opt.cuda:
        C = C.float().cuda(opt.devices[0])
        criterion_BCE = criterion_BCE.cuda(opt.devices[0])
        criterion_KL = criterion_KL.cuda(opt.devices[0])
        model = model.cuda(opt.devices[0])
        model_truncation = model_truncation.cuda(opt.devices[0])
        AdditionalLayer = AdditionalLayer.cuda(opt.devices[0])
        cudnn.benchmark = True
    
    if (opt.mode == "Train") | (opt.mode == "train"):
        train_truncation(model_truncation, AdditionalLayer, criterion_BCE, train_set, opt, C)
    elif (opt.mode == "Test") | (opt.mode == "test"):
        test_truncation(model_truncation, AdditionalLayer, criterion_KL, test_set, opt, C)


def main(opt):
    # prepare
    initialize(opt)
    train_set, test_set = get_dataset(opt)

    C = get_distance_matrix(opt.dataset_name)
    criterion_BCE = nn.BCELoss()
    model = load_model(opt, opt.num_classes)
    if opt.cuda:
        C = C.float().cuda(opt.devices[0])
        criterion_BCE = criterion_BCE.cuda(opt.devices[0])
        model = model.cuda(opt.devices[0])
        cudnn.benchmark = True        

    if (opt.mode == "Train") | (opt.mode == "train"):
        train(model, criterion_BCE, train_set, test_set, opt)
    elif (opt.mode == "Test") | (opt.mode == "test"):
        _ = test(model, criterion_BCE, test_set, opt, C)


if __name__ == "__main__":
    # parse options 
    op = Options()
    opt = op.parse()

    if opt.truncation != "":
        main_truncation(opt)
    else:
        main(opt)