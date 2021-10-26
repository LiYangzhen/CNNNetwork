from MyDataset import MyDataset
from torchvision import datasets, transforms

IMG_DIR = r"C:\Users\LYZ\Documents\Code\CNNNetwork\train"  # 在此处修改为测试图片的地址

train_data = MyDataset('train.csv', transform=transforms.ToTensor())
