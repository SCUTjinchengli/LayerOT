import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
from torchvision import transforms
from torch.utils.data import DataLoader


def PrepareData(dataset_dir, dataset_name, mode):
    if dataset_name == 'VOC2012':
        dic_path = dataset_dir + '/' + dataset_name + '/ImageSets/Main'
    else:
        dic_path = dataset_dir + '/' + dataset_name + mode + '/ImageSets/Main'

    object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

    image_label_list = []
    files = os.listdir(dic_path)

    for obj in object_categories:
        if mode in ['Train', 'train']:
            file = (obj + '_' + 'trainval.txt')
        else:
            file = (obj + '_' + 'test.txt')
        if (file in files):
            txt_list = open(dic_path + '/' + file).readlines()
            images_len = len(txt_list)  # for train, it is 5011
            image_label_list.append(txt_list)


    assert (len(image_label_list) == 20)

    images_list = []
    images_buffer = []
    labels = np.zeros((len(image_label_list), images_len), dtype = int) #for train, 20 * 5011

    # constructor
    for i, txt_list in enumerate(image_label_list):
        if i == 0:
            for j, line in enumerate(txt_list):
                images_list.append(line.split()[0])
                labels[i][j] = line.split()[-1]
        else:
            for j, line in enumerate(txt_list):
                images_buffer.append(line.split()[0])
                labels[i][j] = line.split()[-1]

    # write image_label_list.txt
    labels = labels.transpose() #for train, 5011 * 20
    labels[labels < 1] = 0
    labels[labels >=1] = 1
    if dataset_name == 'VOC2012':
        images_path = [((dataset_dir + '/' + dataset_name + '/JPEGImages/') + str(element).zfill(6) + '.jpg') for element in images_list]
        wirte_path = dataset_dir + '/' + dataset_name + '/image_label_list.txt'
    else:
        images_path = [((dataset_dir + '/' + dataset_name + mode + '/JPEGImages/') + str(element).zfill(6) + '.jpg') for element in images_list]
        wirte_path = dataset_dir + '/' + dataset_name + mode + '/image_label_list.txt'
    
    with open((wirte_path), 'w') as f:
        for i, line in enumerate(images_path):
            f.write(str(line) + ' ')
            for j, num in enumerate(labels[i].squeeze()):
                if j == (labels[i].squeeze().shape[0] - 1):
                    f.write(str(num))
                else:
                    f.write(str(num) + ' ')
            f.write('\n')


def make_dataset(image_list):
    if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
    else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def open_image(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class BaseDataset(data.Dataset):
    def __init__(self, image_list, transform=None, target_transform=None):
        super(BaseDataset, self).__init__()

        imgs = make_dataset(image_list)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = open_image(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self):
        return len(self.imgs)

class VOC2007DataLoader():
    def __init__(self, opt):
        self.opt = opt
        path_train = opt.dataset_dir + "/" + opt.dataset_name + "train" + "/" + "image_label_list.txt"
        path_test = opt.dataset_dir + "/" + opt.dataset_name + "test" + "/" + "image_label_list.txt"
        print(path_train)
        print(path_test)

        if not os.path.exists(path_train):
            PrepareData(opt.dataset_dir, opt.dataset_name, "train")
        if not os.path.exists(path_test):
            PrepareData(opt.dataset_dir, opt.dataset_name, "test")

        self.train_set = BaseDataset(open(path_train).readlines(), \
            transform=transforms.Compose([
                transforms.Resize(opt.load_size),
                transforms.RandomResizedCrop(opt.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            )
        
        self.test_set = BaseDataset(open(path_test).readlines(), \
            transform=transforms.Compose([
                transforms.Resize(opt.load_size),
                transforms.CenterCrop(opt.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            )

    def GetTrainSet(self):
        return self._DataLoader(self.train_set)

    def GetTestSet(self):
        return self._DataLoader(self.test_set)

    def _DataLoader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=int(self.opt.load_thread),
            pin_memory=self.opt.cuda,
            drop_last=False)
        return dataloader    


def COCOPrepareData(dataset_dir, dataset_name, mode):
    
    img_path = dataset_dir + "/" + dataset_name + "/" + "coco_{}_imglist.txt".format(mode)
    label_path = dataset_dir + "/" + dataset_name + "/" + "coco_{}_label.txt".format(mode)

    img_list = []
    num_labels_per_image = []
    with open(img_path) as f:
        for line in f.readlines():
            img_list.append(line.split()[0])
            num_labels_per_image.append(float(line.split()[-1]))

    label_list = []
    with open(label_path) as f:
        for line in f.readlines():
            label_list.append(line)

    assert len(img_list) == len(label_list)

    with open(dataset_dir + "/" + dataset_name + "/" + \
                "coco_{}_img_label_list.txt".format(mode), 'w') as f:
        for i in range(len(img_list)):
            if num_labels_per_image[i] > 0.:
                buffer = np.array([int(la) for la in label_list[i].split()])
                if buffer.sum() == 0:
                    continue
                else:
                    f.write(dataset_dir + "/" + dataset_name + "/" + img_list[i] + " " + label_list[i])



class MSCOCODataLoader():
    def __init__(self, opt):
        self.opt = opt

        if os.path.exists(opt.dataset_dir + "/" + "coco-txt" + "/" + "coco_train_img_label_list.txt"):
            path_train = opt.dataset_dir + "/" + "coco-txt" + "/" + "coco_train_img_label_list.txt"
            path_test = opt.dataset_dir + "/" + "coco-txt" + "/" + "coco_test_img_label_list.txt"
        else:
            path_train = opt.dataset_dir + "/" + opt.dataset_name + "/" + "coco_train_img_label_list.txt"
            path_test = opt.dataset_dir + "/" + opt.dataset_name + "/" + "coco_test_img_label_list.txt"
        
        if not os.path.exists(path_train):
            COCOPrepareData(opt.dataset_dir, opt.dataset_name, "train")
        if not os.path.exists(path_test):
            COCOPrepareData(opt.dataset_dir, opt.dataset_name, "test")
        print(path_train)
        print(path_test)

        self.train_set = BaseDataset(open(path_train).readlines(), \
            transform=transforms.Compose([
                transforms.Resize(opt.load_size),
                transforms.RandomResizedCrop(opt.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            )

        self.test_set = BaseDataset(open(path_test).readlines(), \
            transform=transforms.Compose([
                transforms.Resize(opt.load_size),
                transforms.CenterCrop(opt.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            )

    def GetTrainSet(self):
        return self._DataLoader(self.train_set)

    def GetTestSet(self):
        return self._DataLoader(self.test_set)

    def _DataLoader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=int(self.opt.load_thread),
            pin_memory=self.opt.cuda,
            drop_last=False)
        return dataloader