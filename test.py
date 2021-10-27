import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from cnn import CNN
from tools import accuracy, get_variable

batch_size = 16
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test():
    test_dataset = MyDataset('test', transforms.ToTensor(), transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    net = CNN()
    if torch.cuda.is_available():
        net = net.cuda()
    net.load_state_dict(torch.load('cnn.pkl'))
    val_accuracy = []
    net.eval()
    for (data, target) in test_loader:
        output = net(get_variable(data))
        accuracies = accuracy(output, get_variable(target))
        val_accuracy.append(accuracies)

    val_r = (sum([tup[0] for tup in val_accuracy]), sum([tup[1] for tup in val_accuracy]))
    print('测试集准确率: {:.2f}%'.format(100. * val_r[0] / val_r[1]))
