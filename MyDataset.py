import csv
import os

import numpy
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.dir_list = os.listdir(root)
        self.label_cnt = len(self.dir_list)

        img_list = []

        for path in self.dir_list:
            sublist = os.listdir(os.path.join(self.root, path))
            for sub_path in sublist:
                img_list.append(sub_path)
        self.img_list = img_list
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        label = index % 12
        path = os.path.join(self.root, self.dir_list[label], self.img_list[index])
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label

    def __len__(self):
        return len(self.img_list)
