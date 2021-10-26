import csv
import torch
from PIL.Image import Image

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None):
        super(MyDataset, self).__init__()

        y = []
        x = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header_row = next(reader)
            # print(header_row)
            for row in reader:
                y.append(int(row[0]))
                x.append(list(map(int, row[1:])))
        self.data = x
        self.label = torch.LongTensor(y)
        self.transform = transform
        self.target_transform = target_transform


def __getitem__(self, index):
    lable = self.lable[index]
    # 读取图像文件
    img_as_img = self.data[index]
    if self.transform is not None:
        img_as_img = self.transform(img_as_img)  # 是否进行transform

    return img_as_img, lable  # 返回每一个index对应的图片数据和对应的label


def __len__(self):
    return len(self.data)
