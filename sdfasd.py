from torch.utils.data import DataLoader
from torchvision import transforms

from MyDataset import MyDataset

train_dataset = MyDataset('train.csv', transforms.ToTensor(), transforms.ToTensor())
test_dataset = MyDataset('test_data.csv', transforms.ToTensor(), transforms.ToTensor())

images =DataLoader(train_dataset, bat3ch_size=6, shuffle=True, drop_last=False, num_workers=2)