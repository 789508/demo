import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    """ return given network
    """
    if args.class_num:
        class_num = args.class_num

    if args.net == 'vgg16':
        from model.VGGNet import VGG16
        net = VGG16(class_num=class_num)
    elif args.net == 'vgg19':
        from model.VGGNet import VGG19
        net = VGG19(class_num=class_num)
    elif args.net == 'alexnet':
        from model.AlexNet import alexnet
        net = alexnet(class_num=class_num)
    elif args.net == 'densenet121':
        from model.DenseNet import DenseNet121
        net = DenseNet121(class_num=class_num)
    elif args.net == 'densenet169':
        from model.DenseNet import DenseNet169
        net = DenseNet169(class_num=class_num)
    elif args.net == 'densenet201':
        from model.DenseNet import DenseNet201
        net = DenseNet201(class_num=class_num)
    elif args.net == 'densenet264':
        from model.DenseNet import DenseNet264
        net = DenseNet264(class_num=class_num)
    elif args.net == 'googlenet':
        from model.GoogleNet import googlenet
        net = googlenet(class_num=class_num)
    elif args.net == 'inceptionv1':
        from model.InceptionV1 import inceptionv1
        net = inceptionv1(class_num=class_num)
    elif args.net == 'inceptionv2':
        from model.InceptionV2 import inceptionv2
        net = inceptionv2(class_num=class_num)
    elif args.net == 'inceptionv3':
        from model.InceptionV3 import inceptionv3
        net = inceptionv3(class_num=class_num)
    elif args.net == 'inceptionv4':
        from model.InceptionV4 import inceptionv4
        net = inceptionv4(class_num=class_num)
    elif args.net == 'resnet50':
        from model.ResNet import ResNet50
        net = ResNet50(class_num=class_num)
    elif args.net == 'resnet101':
        from model.ResNet import ResNet101
        net = ResNet101(class_num=class_num)
    elif args.net == 'resnet152':
        from model.ResNet import ResNet152
        net = resnet50(class_num=class_num)
    elif args.net == 'preactresnet18':
        from model.preactresnet import preactresnet18
        net = preactresnet18(class_num=class_num)
    elif args.net == 'preactresnet34':
        from model.preactresnet import preactresnet34
    elif args.net == 'preactresnet50':
        net = preactresnet34(class_num=class_num)
        from model.preactresnet import preactresnet50
        net = preactresnet50(class_num=class_num)
    elif args.net == 'preactresnet101':
        from model.preactresnet import preactresnet101
        net = preactresnet101(class_num=class_num)
    elif args.net == 'preactresnet152':
        from model.preactresnet import preactresnet152
        net = preactresnet152(class_num=class_num)
    elif args.net == 'resnext50':
        from model.ResNeXt import ResNeXt50
        net = ResNeXt50(class_num=class_num)
    elif args.net == 'resnext101':
        from model.ResNeXt import ResNeXt101
        net = ResNeXt101(class_num=class_num)
    elif args.net == 'resnext152':
        from model.ResNeXt import ResNeXt152
        net = ResNeXt152(class_num=class_num)
    elif args.net == 'mobilenet':
        from model.mobilNet import mobilenet
        net = mobilenet(class_num=class_num)
    elif args.net == 'mobilenetv2':
        from model.mobileNetv2 import mobilenetv2
        net = mobilenetv2(class_num=class_num)
    elif args.net == 'nasnet':
        from model.NasNet import nasnet
        net = nasnet(class_num=class_num)
    elif args.net == 'attention56':
        from model.attention import attention56
        net = attention56(class_num=class_num)
    elif args.net == 'attention92':
        from model.attention import attention92
        net = attention92(class_num=class_num)
    elif args.net == 'seresnet18':
        from model.SeNet import seresnet18
        net = seresnet18(class_num=class_num)
    elif args.net == 'seresnet34':
        from model.SeNet import seresnet34
        net = seresnet34(class_num=class_num)
    elif args.net == 'seresnet50':
        from model.SeNet import seresnet50
        net = seresnet50(class_num=class_num)
    elif args.net == 'seresnet101':
        from model.SeNet import seresnet101
        net = seresnet101(class_num=class_num)
    elif args.net == 'seresnet152':
        from model.SeNet import seresnet152
        net = seresnet152(class_num=class_num)
    elif args.net == 'rirnet':
        from model.RiR import resnet_in_resnet
        net = resnet_in_resnet(class_num=class_num)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
