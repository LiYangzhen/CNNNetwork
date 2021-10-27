import csv
import os

import numpy
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        self.path = path
        self.dirlist = os.listdir(path)
        self.label_cnt = len(self.dirlist)

        imglist = []

        for path in self.dirlist:
            sublist = os.listdir(os.path.join(self.path, path))
            for subpath in sublist:
                imglist.append(subpath)
        self.imglist = imglist

        # y = []
        # x = []
        # with open(csv_path, 'r') as f:
        #     reader = csv.reader(f)
        #     header_row = next(reader)
        #     # print(header_row)
        #     for row in reader:
        #         y.append(int(row[0]))
        #         a = list(map(float, row[1:]))
        #         x.append(list(map(int, a)))
        # self.data = x
        # self.label = y
        self.transform = transform
        self.target_transform = target_transform
        # print(len(self.label))

    def __getitem__(self, index):
        label = index % 12
        path = os.path.join(self.path, self.dirlist[label], self.imglist[index])
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform

        return img, label

        # label = self.label[index]
        # # 读取图像文件
        # image = self.data[index]
        # if self.transform is not None:
        #     image = self.transform(numpy.array(image))  # 是否进行transform
        #
        # # if self.target_transform is not None:
        # #     label = self.target_transform(label)
        # return image, label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return len(self.imglist)
