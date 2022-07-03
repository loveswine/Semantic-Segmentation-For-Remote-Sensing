import cv2
import numpy as np
import os
from tqdm import tqdm
import random, time

from multiprocessing import Pool

patch_size_list = [512, 768, 1024, 1280]
input_dir_list = ['rgb', 'labelrgb']
patch_size = 512
cutsize_h = patch_size
cutsize_w = patch_size
stride = 256
input_path = r"../../image/test"
output_path = r"../../dataset/test/{}".format(patch_size)

if not os.path.exists(output_path):
    os.makedirs(output_path)
output_rgb_path = os.path.join(output_path, 'rgb')
if not os.path.exists(output_rgb_path):
    os.mkdir(output_rgb_path)
output_label_path = os.path.join(output_path, 'labelrgb')
if not os.path.exists(output_label_path):
    os.mkdir(output_label_path)


def RandomCut(filename):
    image_path = os.path.join(input_path, input_dir_list[0], filename)
    label_path = os.path.join(input_path, input_dir_list[1], filename)
    img = cv2.imread(image_path)
    label = cv2.imread(label_path)
    assert img.shape == label.shape
    h, w = img.shape[0], img.shape[1]
    
    h_pad_cutsize = ((h - cutsize_h) // stride + 1) * stride + cutsize_h
    w_pad_cutsize = ((w - cutsize_w) // stride + 1) * stride + cutsize_w
    img = cv2.copyMakeBorder(img, 0, h_pad_cutsize - h, 0, w_pad_cutsize - w, cv2.BORDER_CONSTANT, 0)
    label = cv2.copyMakeBorder(label, 0, h_pad_cutsize - h, 0, w_pad_cutsize - w, cv2.BORDER_CONSTANT, 0)
    # 对大图进行padding
    # h_pad_cutsize = h if (h // cutsize_h == 0) else (h // cutsize_h + 1) * cutsize_h
    # w_pad_cutsize = w if (w // cutsize_w == 0) else (w // cutsize_w + 1) * cutsize_w
    # img = cv2.copyMakeBorder(img, 0, h_pad_cutsize - h, 0, w_pad_cutsize - w, cv2.BORDER_CONSTANT, 0)
    # label = cv2.copyMakeBorder(label, 0, h_pad_cutsize - h, 0, w_pad_cutsize - w, cv2.BORDER_CONSTANT, 0)
    
    # h_pad_stride, w_pad_stride = h_pad_cutsize + stride, w_pad_cutsize + stride
    # img = cv2.copyMakeBorder(img, stride // 2, stride // 2, stride // 2, stride // 2, cv2.BORDER_CONSTANT, 0)
    # label = cv2.copyMakeBorder(label, stride // 2, stride // 2, stride // 2, stride // 2, cv2.BORDER_CONSTANT, 0)
    
    index = 0
    for i in range(0, h_pad_cutsize // stride - 1):
        for j in range(0, w_pad_cutsize // stride - 1):
            index = index + 1
            topleft_y = i * stride
            topleft_x = j * stride
            img_cut = img[topleft_y:topleft_y + cutsize_h, topleft_x:topleft_x + cutsize_w, :]
            label_cut = label[topleft_y:topleft_y + cutsize_h, topleft_x:topleft_x + cutsize_w, :]
            # 检查大小
            if img_cut.shape[:2] != (cutsize_h, cutsize_w):
                print(topleft_x, topleft_y, img_cut.shape)
            # 过滤掉全部是黑色图片
            if np.sum(img_cut) == 0:
                continue
            # 过滤掉无样本标签
            if np.sum(label_cut) == 0 and random.random() < 0.4:
                continue
            
            image_save_path = os.path.join(output_rgb_path, filename.replace('.png', '_%03d.png' % index))
            label_save_path = os.path.join(output_label_path, filename.replace('.png', '_label_%03d.png' % index))
            
            cv2.imwrite(image_save_path, img_cut)
            cv2.imwrite(label_save_path, label_cut)


if __name__ == "__main__":
    filenames = os.listdir(os.path.join(input_path, input_dir_list[0]))
    # t1=time.time()
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(RandomCut, filenames), total=len(filenames), desc='Cuting progress'))
    
    # for filename in tqdm(filenames):
    #
    #     RandomCut(filename)
    #
    # print(time.time()-t1)
