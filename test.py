# -*- coding: utf-8 -*-
"""
@author: xqxqxxq
"""
import argparse
import time
import os
import json
from dataset import RSDataset
import sync_transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from models.deeplabv3_version_1.deeplabv3 import DeepLabV3 as model1
from models.deeplabv3_version_2.deeplabv3 import DeepLabV3 as model2
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import torchvision
from torchvision import transforms
from palette import colorize_mask
from PIL import Image
from collections import OrderedDict
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset-name', type=str, default='china south')
    parser.add_argument('--test-data-root', type=str, default=r'./dataset/test/512')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='batch size for training (default:16)')
    
    # output_save_path
    parser.add_argument('--experiment-start-time', type=str,
                        default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    parser.add_argument('--save-pseudo-data-path', type=str, default='./pseudo_data')
    # augmentation
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')
    # model
    parser.add_argument('--model', type=str, default='deeplabv3_version_1', help='model name')
    parser.add_argument("--model-path", type=str, default=r"./workdirectory40/epoch_115_acc_0.93990_kappa_0.91563.pth")
    parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--n-blocks', type=str, default='3, 4, 23, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='')
    parser.add_argument('--multi-grids', type=str, default='1, 1, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='6, 12, 18', help='')
    parser.add_argument('--deeplabv3-no-global-pooling', action='store_true', default=False)
    parser.add_argument('--deeplabv3-use-deformable-conv', action='store_true', default=False)
    parser.add_argument('--no-syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    
    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=1, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=4)
    # validation
    parser.add_argument('--eval', action='store_true', default=True, help='evaluation only')
    
    parser.add_argument('--best-kappa', type=float, default=0)
    
    args = parser.parse_args([])
    directory = "./workdirectory"
    args.directory = directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if args.use_cuda:
        print('Numbers of GPUs:', args.num_GPUs)
    else:
        print("Using CPU")
    return args


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Tester(object):
    def __init__(self, args):
        self.args = args
        resize_scale_range = [float(scale) for scale in args.resize_scale_range.split(',')]
        sync_transform = sync_transforms.Compose([
            sync_transforms.RandomScale(args.base_size, args.crop_size, resize_scale_range),
            sync_transforms.RandomFlip(args.flip_ratio)
        ])
        self.resore_transform = transforms.Compose([
            DeNormalize([.485, .456, .406], [.229, .224, .225]),
            transforms.ToPILImage()
        ])
        self.visualize = transforms.Compose([transforms.ToTensor()])
        
        print(args.test_data_root)
        self.test_dataset = RSDataset(root=args.test_data_root, mode='test', sync_transforms=sync_transform)
        
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=args.test_batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      drop_last=True)
        print('class names {}'.format(self.test_dataset.class_names))
        print('Test samples {}'.format(len(self.test_dataset)))
        
        self.num_classes = len(self.test_dataset.class_names)
        print("类别数：", self.num_classes)
        
        self.criterion = nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-1).cuda()
        
        n_blocks = args.n_blocks
        n_blocks = [int(b) for b in n_blocks.split(',')]
        atrous_rates = args.deeplabv3_atrous_rates
        atrous_rates = [int(s) for s in atrous_rates.split(',')]
        multi_grids = args.multi_grids
        multi_grids = [int(g) for g in multi_grids.split(',')]
        
        if args.model == 'deeplabv3_version_1':
            model = model1(num_classes=self.num_classes)  # dilate_rate=[6,12,18]
            # resume
        
        if args.model == 'deeplabv3_version_2':
            model = model2(num_classes=self.num_classes,
                           n_blocks=n_blocks,
                           atrous_rates=atrous_rates,
                           multi_grids=multi_grids,
                           output_stride=args.output_stride)
        state_dict = torch.load(args.model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        if args.use_cuda:
            model = model.cuda()
            self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
        self.save_pseudo_data_path = args.save_pseudo_data_path
    
    def validating(self, epoch):
        self.model.eval()  # 把module设成预测模式，对Dropout和BatchNorm有影响
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.test_loader)
        with torch.no_grad():
            for index, data in enumerate(tbar):
                # assert data[0].size()[2:] == data[1].size()[1:]
                imgs = Variable(data[0])
                masks = Variable(data[1])
                
                if self.args.use_cuda:
                    imgs = imgs.cuda()
                    masks = masks.cuda()
                
                predict_1 = self.model(imgs)

                predict_2 = self.model(torch.flip(imgs, [-1]))
                predict_2 = torch.flip(predict_2, [-1])

                predict_3 = self.model(torch.flip(imgs, [-2]))
                predict_3 = torch.flip(predict_3, [-2])

                predict_4 = self.model(torch.flip(imgs, [-1, -2]))
                predict_4 = torch.flip(predict_4, [-1, -2])

                predict_list = (predict_1 + predict_2 + predict_3 + predict_4)

                _, preds = torch.max(predict_list, 1)
                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
                
                conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                    label=masks.flatten(),
                                                    num_classes=self.num_classes)
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = metric.evaluate(conf_mat)
        
        table = PrettyTable(["序号", "名称", "acc", "IoU"])
        for i in range(self.num_classes):
            table.add_row([i, self.test_dataset.class_names[i], val_acc_per_class[i], val_IoU[i]])
        print(table)
        print("test_acc:", val_acc)
        print("test_mean_IoU:", val_mean_IoU)
        print("test_kappa:", val_kappa)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    tester = Tester(args)
    tester.validating(epoch=0)
