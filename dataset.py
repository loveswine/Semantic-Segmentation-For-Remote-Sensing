from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
import torch


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img)).long()


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
mask_transform = MaskToTensor()


class RSDataset(Dataset):
    def __init__(self, root=None, mode=None, sync_transforms=None):
        # 数据相关
        self.class_names = ['background', '草地', '道路', '建筑', '水体']
        self.mode = mode
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []
        
        img_dir = root + '/rgb'
        mask_dir = root + '/label'
        
        for img_filename in os.listdir(mask_dir):
            img_mask_pair = (os.path.join(img_dir, img_filename.replace('_label_', '_')),
                             os.path.join(mask_dir, img_filename))  # 原图标签
            self.sync_img_mask.append(img_mask_pair)
        print('path inspect',self.sync_img_mask[0])
        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")
    
    def __getitem__(self, index):
        img_path, mask_path = self.sync_img_mask[index]
        img = Image.open(img_path).convert('RGB')
        # img=Image.fromarray(cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB))
        mask = Image.open(mask_path).convert('L')
        # transform
        if self.sync_transform is not None:
            img, mask = self.sync_transform(img, mask)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask
    
    def __len__(self):
        return len(self.sync_img_mask)
    
    def classes(self):
        return self.class_names


if __name__ == "__main__":
    pass
