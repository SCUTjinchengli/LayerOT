import os
import copy
import numpy as np
import logging
import collections
from PIL import Image
import torch
import math

class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= (pos_count + 1e-10)
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1


def tensor2im(image_tensor, mean, std, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = image_numpy.transpose(1, 2, 0)
    image_numpy *= std
    image_numpy += mean
    image_numpy *= 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path):
    if os.path.exists(path):
        os.system('rm -rf ' + path)

def print_loss(loss_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        logging.info("[ %s Loss ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Loss ] of Epoch %d Batch %d" % (label, epoch, batch_iter))
    logging.info("---- loss:  %f" % (loss_list))

def print_accuracy(accuracy_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        logging.info("[ %s Accu ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Accu ] of Epoch %d Batch %d" %(label, epoch, batch_iter))
    
    for index, item in enumerate(accuracy_list):
        for top_k, value in item.items():
            logging.info("----Attribute %d Top%d: %f" %(index, top_k, value["ratio"]))

def opt2file(opt, dst_file):
    args = vars(opt) 
    with open(dst_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print("%s: %s" %(str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('-------------- End ----------------')

def load_label(label_file):
    rid2name = list()   # rid: real id, same as the id in label.txt
    id2rid = list()     # id: number from 0 to len(rids)-1 corresponding to the order of rids
    rid2id = list()     
    with open(label_file) as l:
        rid2name_dict = collections.defaultdict(str)
        id2rid_dict = collections.defaultdict(str)
        rid2id_dict = collections.defaultdict(str)
        new_id = 0 
        for line in l.readlines():
            line = line.strip('\n\r').split(';')
            if len(line) == 3: # attr description
                if len(rid2name_dict) != 0:
                    rid2name.append(rid2name_dict)
                    id2rid.append(id2rid_dict)
                    rid2id.append(rid2id_dict)
                    rid2name_dict = collections.defaultdict(str)
                    id2rid_dict = collections.defaultdict(str)
                    rid2id_dict = collections.defaultdict(str)
                    new_id = 0
                rid2name_dict["__name__"] = line[2]
                rid2name_dict["__attr_id__"] = line[1]
            elif len(line) == 2: # attr value description
                rid2name_dict[line[0]] = line[1]
                id2rid_dict[new_id] = line[0]
                rid2id_dict[line[0]] = new_id
                new_id += 1
        if len(rid2name_dict) != 0:
            rid2name.append(rid2name_dict)
            id2rid.append(id2rid_dict)
            rid2id.append(rid2id_dict)
    return rid2name, id2rid, rid2id


def chebyshev(real_label, predicted_label):
    temp = torch.abs(predicted_label - real_label)
    temp2, _ = torch.max(temp, dim=1, keepdim=True)

    distance = torch.mean(temp2, dim=0, keepdim=True)

    return distance # return a 1x1 tensor