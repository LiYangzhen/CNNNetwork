import csv

from torch.utils.data import Dataset


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
                a = list(map(float, row[1:]))
                x.append(list(map(int, a)))
        self.data = x
        self.label = y
        self.transform = transform
        self.target_transform = target_transform
        # print(len(self.label))


def __getitem__(self, index):
    label = self.label[index]
    # 读取图像文件
    image = self.data[index]
    if self.transform is not None:
        image = self.transform(image)  # 是否进行transform
    if self.target_transform is not None:
        label = self.target_transform(label)
    return image, label  # 返回每一个index对应的图片数据和对应的label


def __len__(self):
    return len(self.label)
