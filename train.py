import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from tqdm import tqdm

from MyDataset import MyDataset
from cnn import CNN
from tools import accuracy, get_variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 64
learning_rate = 0.001
num_epochs = 5


def train():
    train_dataset = MyDataset('train', transforms.ToTensor(), transforms.ToTensor())
    test_dataset = MyDataset('test', transforms.ToTensor(), transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 根据采样器来定义加载器，然后加载数据
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    net = CNN()
    if torch.cuda.is_available():
        net = net.cuda()

    # 选择损失函数和优化方法
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    record = []  # 记录训练集和验证集上错误率的list
    max_rate = 0
    bar = tqdm(total=len(train_loader) * num_epochs)
    bar.set_description("Epoch[0/" + str(num_epochs) + "]")

    for epoch in range(num_epochs):
        train_accuracy = []
        bar.set_description("Epoch[" + str(epoch + 1) + "/" + str(num_epochs) + "]")
        for i, (images, labels) in enumerate(train_loader):
            bar.update(1)
            net.train()
            images = get_variable(images)
            labels = get_variable(labels)
            outputs = net(images)

            optimizer.zero_grad()
            loss = loss_func(outputs, labels)

            loss.backward()  # 反向传播，自动计算每个节点的锑度至
            optimizer.step()

            accuracies = accuracy(outputs, labels)
            train_accuracy.append(accuracies)

            net.eval()
            val_accuracy = []
            for (data, target) in test_loader:
                output = net(get_variable(data))  # 完成一次前馈计算过程，得到目前训练得到的模型net在校验数据集上的表现
                accuracies = accuracy(output, get_variable(target))  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                val_accuracy.append(accuracies)

            train_r = (sum([tup[0] for tup in train_accuracy]), sum([tup[1] for tup in train_accuracy]))
            val_r = (sum([tup[0] for tup in val_accuracy]), sum([tup[1] for tup in val_accuracy]))

            tmp_loss = "{:.6f}".format(loss.item())
            tmp_train = "{:.2f}%".format(100. * train_r[0] / train_r[1])
            tmp_test = "{:.2f}%".format(100. * val_r[0] / val_r[1])
            bar.set_postfix(Loss=tmp_loss, Train_Accuracy=tmp_train, Test_Accuracy=tmp_test)
            # print('Epoch [{}/{}] [{}/{}]\tLoss: {:.6f}\t训练集准确率: {:.2f}%\t验证集准确率: {:.2f}%'.format(
            #     epoch + 1, num_epochs, i * batch_size, len(train_loader.dataset),
            #     loss.item(),
            #     100. * train_r[0] / train_r[1],
            #     100. * val_r[0] / val_r[1]))

            record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

            if val_r[0] / val_r[1] > max_rate:
                max_rate = val_r[0] / val_r[1]

            if val_r[0] / val_r[1] > 0.9875:
                max_rate = val_r[0] / val_r[1]
                torch.save(net.state_dict(), 'cnn.pkl')

    bar.close()
    plt.figure(figsize=(10, 7))
    record2 = torch.Tensor(record).numpy().tolist()
    plt.plot(record2)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy rate')
    plt.gca().invert_yaxis()
    plt.show()
    print(max_rate)
