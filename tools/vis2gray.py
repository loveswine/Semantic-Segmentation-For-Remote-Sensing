import json, os
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio

'''
总体思路就是逐一读彩色的label，通过Palette.json文件找到映射关系
例如：
{ "0": [0,0,0],
  "1": [0,200,0],
  "2": [150,250,0],
  "3": [150,200,150],
  "4": [200,0,200],
  "5": [150,0,250],
  "6": [150,150,250],
  "7": [250,200,0],
  "8": [200,200,0],
  "9": [200,0,0],
  "10": [250,0,150],
  "11": [200,150,150],
  "12": [250,150,150],
  "13": [0,0,200],
  "14": [0,150,200],
  "15": [0,200,250]
}
那么，如果读的像素是[0,200,0]，则将1作为相应位置的灰度值，以此类推。

{ "0": [0,0,0],
  "1": [255,0,0],
  "2": [0,255,0],
  "3": [0,255,255],
  "4": [255,255,0],
  "5": [0,0,255],
}
['其他类别',
'建筑',
'田地',
'森林',
'草地',
'水体']


'''


def main(path):
    palette = {"0": [0, 0, 0],
               "1": [255, 0, 0],
               "2": [0, 255, 0],
               "3": [0, 255, 255],
               "4": [255, 255, 0],
               "5": [0, 0, 255]}
    for pic in os.listdir(path):
        if 'label' in pic:
            print(path + '/' + pic)
            label=cv2.imread(path + '/' + pic)
  
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            # cv2.namedWindow('pic', 0)
            # cv2.imshow('pic',label[:,:,0])
            # cv2.waitKey()
            label = np.array(label, dtype=np.uint16)
            label[label < 128] = 0
            label[label >= 128] = 255
            label = label[:, :, 0] * 100 + label[:, :, 1] * 10 + label[:, :, 2]

            label_idx = np.zeros(label.shape, dtype=np.uint8)
            
            for key in palette.keys():
                color = palette[key]
                value = color[0] * 100 + color[1] * 10 + color[2]
                label_idx[label == value] = int(key)

            label = Image.fromarray(label_idx)
            label.save(path + '/' + pic[:-4] + '.png')


if __name__ == '__main__':
    train_path = r'D:\BaiduNetdiskDownload\Fine Land-cover5\label'
    main(train_path)
    
    # cv2.namedWindow('pic',0)
    # cv2.imshow('pic',label_idx)
    # cv2.waitKey()
    # val_path = r'E:\yaogan\dataProcess\data\val\vis_label'
    # main(val_path)