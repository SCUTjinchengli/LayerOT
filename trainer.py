import os
import sys
import time
import copy
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from util import util
from util.ot_loss import *
from gensim.models import Word2Vec
from models.model import save_model
from data.dataloader import VOC2007DataLoader, MSCOCODataLoader


def initialize(opt):
    # initialize train or test working dir
    trainer_dir = "trainer_" + opt.name
    opt.model_dir = os.path.join(opt.dir, trainer_dir, "Train") 
    opt.test_dir = os.path.join(opt.dir, trainer_dir, "Test") 
    
    if (opt.mode == "Train") | (opt.mode == "train"):
        if not os.path.exists(opt.model_dir):        
            os.makedirs(opt.model_dir)
        log_dir = opt.model_dir 
        log_path = log_dir + "/train.log"
    if (opt.mode == "Test") | (opt.mode == "test"):
        if not os.path.exists(opt.test_dir):
            os.makedirs(opt.test_dir)
        log_dir = opt.test_dir
        log_path = log_dir + "/test.log"

    # save options to disk
    util.opt2file(opt, log_dir+"/opt.txt")
    
    # log setting 
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)


def get_dataset(opt):
    if opt.dataset_name == 'VOC2007':
        data_loader = VOC2007DataLoader(opt)
        opt.num_classes = 20
    elif opt.dataset_name == 'coco':
        data_loader = MSCOCODataLoader(opt)
        opt.num_classes = 80
    else:
        assert False

    train_set = data_loader.GetTrainSet()
    test_set = data_loader.GetTestSet()

    return train_set, test_set


def get_distance_matrix(dataset_name):
    if dataset_name == 'VOC2007':
        labels = [['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']]
    elif dataset_name == 'coco':
        labels = [['person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse','sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake','chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']]

    # train model
    model = Word2Vec(labels, min_count=1)

    L = len(labels[0])
    label_vec = torch.zeros(L, 100)
    for i in range(L):
        label_vec[i, :] = torch.tensor(model[labels[0][i]])
    C = cost_matrix(label_vec, label_vec, p=2)

    return C


def calculate_distance(output, targets, C, opt):
    # for OT Loss
    output_normal = output / torch.sum(output, dim=1, keepdim=True)
    targets_normal = targets / torch.sum(targets, dim=1, keepdim=True)
    C_expand = C.expand(output.size(0), C.size(0), C.size(1))
    P = normalized_parallel_wasserstein_loss_one_gpu(output_normal, targets_normal, C_expand, opt.reg_entropy, C.size(0), opt.niter_ot, False)
    ot_loss = torch.sum(P * C) / output.size(0)

    # for KL Loss
    eps = 1e-10
    kl_loss = torch.sum(output_normal*torch.log(output_normal/(targets_normal+eps) + eps))
    kl_loss = kl_loss / output.size(0)

    # for chebyshev
    chebyshev_loss = util.chebyshev(targets_normal, output_normal)

    # for JS Loss
    js_output = output / torch.sum(output, dim=1, keepdim=True)
    js_M = (js_output + targets_normal) / 2 + eps
    js_loss = 0.5 * torch.sum(js_output * torch.log(js_output / js_M + eps)) + \
            0.5 * torch.sum(targets_normal * torch.log(targets_normal / js_M + eps))
    js_loss = js_loss / output.size(0)
    assert not np.isnan(js_loss.item())

    return ot_loss, kl_loss, chebyshev_loss, js_loss


def forward_batch(model, criterion, inputs, targets, opt, phase):
    if phase in ["Train", "train"]:
        inputs_var = Variable(inputs, requires_grad=True)
        model.train()
    elif phase in ["Validate", "Test", "test"]:
        with torch.no_grad():
            inputs_var = Variable(inputs)
        model.eval()
        
    # forward
    if opt.cuda:
        if len(opt.devices) > 1:
            output = nn.parallel.data_parallel(model, inputs_var, opt.devices)
        else:
            output = model(inputs_var)
        targets = targets.float().cuda(opt.devices[0])

    else:
        output = model(inputs_var)
    
    loss = criterion(output, targets)
    return output, loss


def forward_batch_truncation(model_truncation, AdditionalLayer, criterion, inputs, targets, opt, phase, C):
    if phase in ["Train", "train"]:
        inputs_var = Variable(inputs, requires_grad=True)
        if opt.truncation != "":
            model_truncation.eval()
            AdditionalLayer.train()
    elif phase in ["Validate", "Test", "test"]:
        with torch.no_grad():
            inputs_var = Variable(inputs)
        model_truncation.eval()
        AdditionalLayer.eval()
    # forward    
    if opt.cuda:
        if len(opt.devices) > 1:
            out_trunc, output_model = nn.parallel.data_parallel(model_truncation, inputs_var, opt.devices)
        else:
            out_trunc, output_model = model_truncation(inputs_var)
        targets = targets.float().cuda(opt.devices[0])
    else:
        out_trunc, output_model = model_truncation(inputs_var)

    output = AdditionalLayer(out_trunc.detach())
    loss = criterion(output, targets)

    return output, loss


def train(model, criterion, train_set, test_set, opt):
    # define optimizer
    optimizer = optim.SGD(model.parameters(),
                          opt.lr, 
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size = opt.lr_decay_in_epoch,
                                          gamma = opt.gamma)

    real_data = torch.FloatTensor(opt.batch_size, opt.input_channel, opt.input_size, opt.input_size)
    if opt.cuda:
        real_data = real_data.cuda()
    
    train_batch_num = len(train_set)
    total_batch_iter = 0

    logging.info("######################Train Model######################")
    for epoch in range(opt.sum_epoch):
        loss_ = 0.
        map_ = 0.        
        epoch_batch_iter = 0
        logging.info("Begin of epoch %d" %(epoch))
        for i, data in enumerate(train_set):
            model.zero_grad()
            inputs, targets = data
            # make -1/0 -> 0
            targets[targets<=0] = 0
            
            real_data.data.resize_(inputs.size()).copy_(inputs)
            output, loss = forward_batch(model, criterion, real_data, targets, opt, "Train")
            loss.backward()
            optimizer.step()

            ap_meter = util.AveragePrecisionMeter(difficult_examples=False)
            ap_meter.reset()
            ap_meter.add(output.cpu().detach(), targets.float().cpu().detach())
            ap_per_class = 100 * ap_meter.value()
            mean_ap = ap_per_class.mean()

            loss_ += loss.cpu().detach().numpy()
            map_ += mean_ap

            epoch_batch_iter += 1
            total_batch_iter += 1
            util.print_loss(loss.item(), "Train", epoch, total_batch_iter)

        loss_ = loss_ / train_batch_num
        map_ = map_ / train_batch_num

        if (epoch+1) % opt.save_epoch_freq == 0:
            logging.info('saving the model at the end of epoch %d, iters %d' %(epoch+1, total_batch_iter))
            save_model(model, opt, epoch+1, best=False) 

        # adjust learning rate 
        scheduler.step()
        lr = optimizer.param_groups[0]['lr'] 
        logging.info('learning rate = %.7f epoch = %d' %(lr,epoch)) 
    logging.info("--------Optimization Done--------")


def train_truncation(model_truncation, AdditionalLayer, criterion, train_set, opt, C):
    # define optimizer
    optimizer = optim.SGD(AdditionalLayer.parameters(),
                          opt.lr, 
                          momentum=opt.momentum, 
                          weight_decay=opt.weight_decay)
    
    # define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size = opt.lr_decay_in_epoch,
                                          gamma = opt.gamma)

    real_data = torch.FloatTensor(opt.batch_size, opt.input_channel, opt.input_size, opt.input_size)
    if opt.cuda:
        real_data = real_data.cuda()
    
    train_batch_num = len(train_set)
    total_batch_iter = 0
    logging.info("######################Train Model######################")
    for epoch in range(opt.sum_epoch):
        
        epoch_batch_iter = 0
        logging.info("Begin of epoch %d" %(epoch))
        for i, data in enumerate(train_set):
            AdditionalLayer.zero_grad()
            inputs, targets = data
            # make -1/0 -> 0
            targets[targets<=0] = 0

            real_data.data.resize_(inputs.size()).copy_(inputs)
            output, loss = forward_batch_truncation(model_truncation, AdditionalLayer, criterion, real_data, targets, opt, "Train", C)
            loss.backward()
            optimizer.step()

            epoch_batch_iter += 1
            total_batch_iter += 1

            # train loss and accuracy
            if total_batch_iter % opt.display_train_freq == 0:
                util.print_loss(loss.item(), "Train", epoch, total_batch_iter)

        if (epoch+1) % opt.save_epoch_freq == 0:
            logging.info('saving the AdditionalLayer at the end of epoch %d, iters %d' %(epoch+1, total_batch_iter))
            save_model(AdditionalLayer, opt, epoch+1, best=False) 

        # adjust learning rate 
        scheduler.step()
        lr = optimizer.param_groups[0]['lr'] 
        logging.info('learning rate = %.7f epoch = %d' %(lr,epoch)) 
    logging.info("--------Optimization Done--------")


def test(model, criterion, test_set, opt, C):
    test_data = torch.FloatTensor(opt.batch_size, opt.input_channel, opt.input_size, opt.input_size)
    if opt.cuda:
        test_data = test_data.cuda()

    test_batch_num = len(test_set)
    # voc2007 4952 / coco 40137 
    if opt.dataset_name == 'VOC2007':
        output = torch.FloatTensor(4952, opt.num_classes)
        target = torch.FloatTensor(4952, opt.num_classes)
    elif opt.dataset_name == 'coco': # test 40137 train 82081
        output = torch.FloatTensor(40137, opt.num_classes)
        target = torch.FloatTensor(40137, opt.num_classes)

    logging.info("####################Test Model###################")
    test_batch_iter = 0
    map_list = []
    ap_per_class_ = torch.FloatTensor(test_batch_num, opt.num_classes)
    
    use_loss = 'bceloss'

    for i, data in enumerate(test_set):
        inputs, targets = data
        test_data.data.resize_(inputs.size()).copy_(inputs)
        # make -1/0 -> 0
        targets[targets<=0] = 0

        test_output, test_loss = forward_batch(model, criterion, test_data, targets, opt, "Test")        
        test_output_cpu = test_output.cpu().detach()
        targets_cpu = targets.cpu().detach()
        for j in range(test_data.size(0)):
            output[i * opt.batch_size + j] = test_output_cpu[j]
            target[i * opt.batch_size + j] = targets_cpu[j]

    ot_loss, kl_loss, chebyshev_loss, js_loss = calculate_distance(output, target, C.cpu(), opt)

    # we name the last layer as layer9 in Resnet18, and so on.
    if opt.model == 'Resnet18':
        truncation = 'layer9'
    elif opt.model == 'VGG16':
        truncation = 'layer13'
    elif opt.model == 'Resnet50':
        truncation = 'layer17'

    with open('%s_%s_%s_ot_kl_chebyshev_js_distribution_distance_test_set.txt'\
    %(opt.dataset_name, use_loss, opt.model), 'a') as f:
        f.writelines("%s,%s,%s,%s,ot_kl_chebyshev_js,%.4f,%.4f,%.4f,%.4f\n"\
        %(opt.dataset_name, use_loss, opt.model, truncation, ot_loss, kl_loss, chebyshev_loss, js_loss))
    return 0


def test_truncation(model_truncation, AdditionalLayer, criterion, test_set, opt, C):
    test_data = torch.FloatTensor(opt.batch_size, opt.input_channel, opt.input_size, opt.input_size)
    if opt.cuda:
        test_data = test_data.cuda()

    test_batch_num = len(test_set)
    logging.info("####################Test Model###################")
    ot_loss_list = []
    kl_loss_list = []
    chebyshev_loss_list = []
    js_loss_list = []
    if opt.dataset_name == 'VOC2007':
        output = torch.FloatTensor(4952, opt.num_classes)
        target_ = torch.FloatTensor(4952, opt.num_classes)
    elif opt.dataset_name == 'coco':
        output = torch.FloatTensor(40137, opt.num_classes)
        target_ = torch.FloatTensor(40137, opt.num_classes)

    use_loss = 'bceloss'

    for i, data in enumerate(test_set):

        inputs, targets = data
        test_data.data.resize_(inputs.size()).copy_(inputs)
        # make -1/0 -> 0
        targets[targets<=0] = 0
        test_output, _ = forward_batch_truncation(model_truncation, AdditionalLayer, criterion, test_data, targets, opt, "Test", C)
                
        for j in range(test_data.size(0)):
            output[i * opt.batch_size + j] = test_output[j].cpu().detach()
            target_[i * opt.batch_size + j] = targets[j].cpu().detach()

    ot_loss, kl_loss, chebyshev_loss, js_loss = calculate_distance(output, target_, C.cpu(), opt)

    with open('%s_%s_%s_ot_kl_chebyshev_js_distribution_distance_test_set.txt'\
    %(opt.dataset_name, use_loss, opt.model), 'a') as f:
        f.writelines("%s,%s,%s,%s,ot_kl_chebyshev_js,%.4f,%.4f,%.4f,%.4f\n"\
        %(opt.dataset_name, use_loss, opt.model, opt.truncation, ot_loss, kl_loss, chebyshev_loss, js_loss))

    logging.info("\n#################Finished Testing################")