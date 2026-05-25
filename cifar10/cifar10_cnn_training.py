import torch
import torchvision
import torchvision.transforms as transforms

# transform.Compose形成了一系列的操作，ToTensor和Normalize
# ToTensor将数据转换成torch.Tensor数据形式（深度学习机器学习中的基础数据形式，可以自动求导或者梯度，可反向传播等）
# 并自动将CIFAR-10中类型：uint8 像素范围：0 ~ 255的原始数据转变为float32数据类型（uint8不能用于反向传播，没有梯度）把数值范围归一化到[0, 1]
# Normalize：Xnorm =（X-mean）/ std，这个0.5只是经验值，用于简化表示（可以变为[-1, 1]），实际上去求可能更好一点
# CIFAR-10 是一个用于计算机视觉入门与基准测试的小型图像数据集,公用的，具体查看readme.html

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform) # default = True
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# cifar10 some of the training images
import matplotlib.pyplot as plt
import numpy as np
# #import sys, os

# # functions to cifar10 an image
def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ ==  '__main__':
# get some random training images
    dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    images, labels = next(dataiter)
    # cifar10 images
    imgshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# model structure
import torch.nn as nn
import torch.nn.functional as F

# set CUDA device
use_cuda = 0
device = torch.device("cuda:0" if torch.cuda.is_available()&use_cuda else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # 第一层卷积和下采样
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积和下采样
        x = self.pool(F.relu(self.conv2(x)))
        # 展平操作：将三维特征图展平为一维向量
        x = x.view(-1, 64 * 5 * 5)
        # 全连接层，进行非线性变换，学习特征之间的复杂关系
        x = F.relu(self.fc1(x))
        x= self.drop(x)
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

net = Net()
net.to(device)

# define loss
import torch.optim as optim

criterion = nn.CrossEntropyLoss()


# define optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



# Train the network
import time
start_time = time.time()


if __name__ == '__main__':

    for epoch in range(40):
        if epoch != 0 and epoch % 20 == 0:
            scheduler.step()
        training_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            #zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize/update
            outputs = net(inputs)
            # Calculating the loss
            loss = criterion(outputs, labels)
            # Calculating the gradient in this process
            loss.backward()
            # optimize based on the gradient and lr
            optimizer.step()

            training_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, training_loss / 2000))
                training_loss = 0.0

    end_time = time.time()
    print ('The time cost for training process is: %7f' % (end_time-start_time))
    print('End Training')

    print('Start Testing')

    correct_total = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
         images, labels = data
         images, labels = images.to(device), labels.to(device)
         outputs = net(images)
         _, predicted = torch.max(outputs.data, 1)
         total += labels.size(0)
         correct_total += (predicted == labels).sum()
         correct_minibatch = (predicted == labels).type(torch.int64)
         for i in range (4):
              label = labels[i]
              class_correct[label] += correct_minibatch[i]
              class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_total / total))
    for i in range(10):
         print ('Accuracy of the class %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))