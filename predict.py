import argparse
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.deeplabv3_version_1.deeplabv3 import DeepLabV3

from torch.autograd import Variable
import torch
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn as nn

from torchvision import transforms
from palette import colorize_mask

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])


def snapshot_forward(model, dataloader, png):
    model.eval()
    for (index, (image, pos_list)) in enumerate(dataloader):
        image = Variable(image).cuda()
        
        predict_1 = model(image)
        
        predict_2 = model(torch.flip(image, [-1]))
        predict_2 = torch.flip(predict_2, [-1])
        
        predict_3 = model(torch.flip(image, [-2]))
        predict_3 = torch.flip(predict_3, [-2])
        
        predict_4 = model(torch.flip(image, [-1, -2]))
        predict_4 = torch.flip(predict_4, [-1, -2])
        
        predict_list = (predict_1 + predict_2 + predict_3 + predict_4)
        
        predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w
        
        batch_size = predict_list.shape[0]  # batch大小
        for i in range(batch_size):
            predict = predict_list[i]
            
            topleft_y, topleft_x, cutsize = pos_list[0][i], pos_list[1][i], pos_list[2][i]
            try:
                png[topleft_y:topleft_y + cutsize, topleft_x:topleft_x + cutsize] = predict
            except:
                print(topleft_y, topleft_x, predict.shape)
        # overlap = colorize_mask(np.squeeze(predict_list,0))
        # overlap.save('./vil/{}.png'.format(index))
    return png


def parse_args():
    parser = argparse.ArgumentParser(description="膨胀预测")
    parser.add_argument('--test-data-root', type=str, default='./image/test/rgb')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=2)
    
    parser.add_argument("--model-path", type=str, default=r"./workdirectory/epoch_115_acc_0.93990_kappa_0.91563.pth")
    parser.add_argument("--pred-path", type=str, default="")
    args = parser.parse_args()
    return args


class Inference_Dataset(Dataset):
    def __init__(self, root_dir, file_name, cutsize, stride, transforms):
        self.transforms = transforms
        self.file_name = file_name
        self.img = cv2.cvtColor(cv2.imread(os.path.join(root_dir, file_name)), cv2.COLOR_BGR2RGB)
        # self.outputpath=os.path.join(root_dir,file_name.split('.')[0])
        # if not os.path.exists(self.outputpath):
        #     os.makedirs(self.outputpath)
        self.cut_img = self.RandomCut(self.img, cutsize, stride)
        
        if (len(self.cut_img)) == 0:
            print("Found 0 data, please check your dataset!")
    
    def RandomCut(self, img, cutsize=512, stride=256):
        h, w = img.shape[0], img.shape[1]
        h_pad_cutsize = ((h - cutsize) // stride + 1) * stride + cutsize
        w_pad_cutsize = ((w - cutsize) // stride + 1) * stride + cutsize
        zeros = (h_pad_cutsize, w_pad_cutsize)  # 填充空白边界，考虑到边缘数据
        self.zeros = np.zeros(zeros, np.uint8)
        
        img = cv2.copyMakeBorder(img, 0, h_pad_cutsize - h, 0, w_pad_cutsize - w, cv2.BORDER_CONSTANT, 0)
        cut_imgs = []
        index = 0
        for i in range(0, h_pad_cutsize // stride - 1):
            for j in range(0, w_pad_cutsize // stride - 1):
                index = index + 1
                topleft_y = i * stride
                topleft_x = j * stride
                img_cut = img[topleft_y:topleft_y + cutsize, topleft_x:topleft_x + cutsize, :]
                # 检查大小
                if img_cut.shape[:2] != (cutsize, cutsize):
                    print(topleft_x, topleft_y, img_cut.shape)
                # save_name=self.file_name.split('.')[0] + '_%003d_%003d_%003d.' % (i, j,index) + self.file_name.split('.')[1]
                # image_save_path = os.path.join(self.outputpath,save_name)
                # cv2.imwrite(image_save_path, img_cut)
                cut_imgs.append((img_cut, topleft_y, topleft_x, cutsize))
        return cut_imgs
    
    def __len__(self):
        return len(self.cut_img)
    
    def __getitem__(self, idx):
        image = self.cut_img[idx][0]
        # print(filename)
        # image_path = os.path.join(self.outputpath, filename)
        # image = np.asarray(Image.open(image_path))  # mode:RGBA
        # image = cv.cvtColor(image, cv.COLOR_RGBA2BGRA)  # PIL(RGBA)-->cv2(BGRA)
        # image = Image.open(image_path).convert('RGB')
        
        image = self.transforms(image)
        
        pos_list = (self.cut_img[idx][1:])
        
        return image, pos_list


def reference():
    args = parse_args()
    files = []
    for file in os.listdir(args.test_data_root):
        if os.path.isfile(os.path.join(args.test_data_root, file)):
            files.append(file)
    cutsize = 512
    stride = 256
    for file in files:
        print(file)
        dataset = Inference_Dataset(root_dir=args.test_data_root, file_name=file, cutsize=cutsize, stride=stride,
                                    transforms=img_transform)
        dataloader = DataLoader(dataset=dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
        
        model = DeepLabV3(num_classes=5)  # dilate_rate=[6,12,18]
        state_dict = torch.load(args.model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
        image = snapshot_forward(model, dataloader, dataset.zeros)
        h, w, _ = dataset.img.shape
        image = image[0:h, 0:w]
        label_save_path = os.path.join(os.path.dirname(args.test_data_root), 'predict', file.split('.')[0] + '.png')
        if not os.path.exists(os.path.join(os.path.dirname(args.test_data_root), 'predict')):
            os.makedirs(os.path.join(os.path.dirname(args.test_data_root), 'predict'))
        
        overlap = colorize_mask(image)
        overlap.save(label_save_path)
        # overlap.show()
        img = Image.fromarray(dataset.img).convert('RGBA')
        overlap = overlap.convert('RGBA')
        blend_image = Image.blend(img, overlap, 0.45)
        blend_image.save('./'+file)
        # blend_image.show()
        plt.axis('off')
        plt.imshow(blend_image)
        
        plt.show()


if __name__ == '__main__':
    reference()
