import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


# 见数据加载器和batch
from MyDataset import MyDataset
from model import CNN

test_dataset = MyDataset('test', transforms.ToTensor(), transforms.ToTensor())


data_loader_test=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)

model = CNN()
model.load_state_dict(torch.load('cnn.pkl'))

X_test, y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_, pred = torch.max(pred, 1)

print("Predict Label is:", [i for i in pred.data])
print("Real Label is :", [i for i in y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1, 2, 0)

plt.imshow(img)
plt.show()