import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

from MyDataset import MyDataset
from model import CNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 64
learning_rate = 0.001
num_epochs = 1


# image_size = 28  # 图像的总尺寸28*28


# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


# return nn.DataParallel(x, device_ids=[0])if torch.cuda.device_count() > 1 else x
# 如果有多个gpu时可以选择上面的语句，例如上面写的时设备0

# train_dataset = torchvision.datasets.ImageFolder('train',
#                                                  transform=transforms.Compose([
#                                                      transforms.Resize((28, 28)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
#                                                      transforms.CenterCrop(28),
#                                                      transforms.ToTensor()])
#                                                  )
#
# test_dataset = torchvision.datasets.ImageFolder('test',
#                                                 transform=transforms.Compose([
#                                                     transforms.Resize((28, 28)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
#                                                     transforms.CenterCrop(28),
#                                                     transforms.ToTensor()])
#                                                 )

train_dataset = MyDataset('train.csv', transforms.ToTensor(), transforms.ToTensor())
test_dataset = MyDataset('test_data.csv', transforms.ToTensor(), transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# indices = range(len(test_dataset))
# indices_val = indices

# 通过下标对验证集和测试集进行采样
# sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
# sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

# 根据采样器来定义加载器，然后加载数据
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=sampler_test)


net = CNN()
if torch.cuda.is_available():
    net = net.cuda()

# 选择损失函数和优化方法
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# print(cnn)

def accuracy(predictions, labels):
    # torch.max的输出：out (tuple, optional维度) – the result tuple of two output tensors (max, max_indices)
    pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    right_num = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return right_num, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


record = []  # 记录训练集和验证集上错误率的list
weights = []  # 每若干步就记录一次卷积核
for epoch in range(num_epochs):
    train_accuracy = []
    for i, (images, labels) in enumerate(train_loader):

        images = get_variable(images)
        labels = get_variable(labels)

        print(images[0])
        print(labels[0])
        outputs = net(images)

        optimizer.zero_grad()
        loss = loss_func(outputs, labels)

        loss.backward()  # 反向传播，自动计算每个节点的锑度至
        optimizer.step()

        accuracies = accuracy(outputs, labels)
        train_accuracy.append(accuracies)

        if i % 100 == 0:
            net.eval()  # 给网络模型做标记，将模型转换为测试模式。
            val_accuracy = []  # 记录校验数据集准确率的容器
            for (data, target) in test_loader:  # 计算校验集上面的准确度

                output = net(get_variable(data))  # 完成一次前馈计算过程，得到目前训练得到的模型net在校验数据集上的表现
                accuracies = accuracy(output, get_variable(target))  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                val_accuracy.append(accuracies)
                # 分别计算在已经计算过的训练集，以及全部校验集上模型的分类准确率

                # train_r为一个二元组，分别记录目前  已经经历过的所有  训练集中分类正确的数量和该集合中总的样本数，
                train_r = (sum([tup[0] for tup in train_accuracy]), sum([tup[1] for tup in train_accuracy]))
                # val_r为一个二元组，分别记录校验集中分类正确的数量和该集合中总的样本数
                val_r = (sum([tup[0] for tup in val_accuracy]), sum([tup[1] for tup in val_accuracy]))

                # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前batch的正确率的平均值
                print('Epoch [{}/{}] [{}/{}]\tLoss: {:.6f}\t训练集准确率: {:.2f}%\t验证集准确率: {:.2f}%'.format(
                    epoch + 1, num_epochs, i * batch_size, len(train_loader.dataset),
                    loss.item(),
                    100. * train_r[0] / train_r[1],
                    100. * val_r[0] / val_r[1]))

                # 将准确率和权重等数值加载到容器中，方便后续处理

                record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

                # weights记录了训练周期中所有卷积核的演化过程。net.conv1.weight就提取出了第一层卷积核的权重
                # clone的意思就是将weight.data中的数据做一个拷贝放到列表中，否则当weight.data变化的时候，列表中的每一项数值也会联动
                '''这里使用clone这个函数很重要'''
                # weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(),
                #                 net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

torch.save(net.state_dict(), 'cnn.pkl')

plt.figure(figsize=(10, 7))
# record2 = torch.Tensor(record)
record2 = torch.Tensor(record).numpy().tolist()
plt.plot(record2)
plt.xlabel('Steps')
plt.ylabel('Error rate')
plt.show()
