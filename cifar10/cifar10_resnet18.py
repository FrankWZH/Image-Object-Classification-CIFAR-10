import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True,
    download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4,
    shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False,
    download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=4,
    shuffle=False, num_workers=2
)

classes = (
    'plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
)

def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imgshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

use_cuda = 0
device = torch.device(
    "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
)
print("Using device:", device)

# 加载 ResNet18
net = models.resnet18(pretrained=False)

# ===== CIFAR-10 改装 =====

# CIFAR-10 图像小（32×32），不用 7×7 卷积
net.conv1 = nn.Conv2d(
    3, 64, kernel_size=3, stride=1, padding=1, bias=False
)

# 去掉 maxpool
net.maxpool = nn.Identity()

# 全连接层：1000 → 10
net.fc = nn.Linear(net.fc.in_features, 10)

net = net.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

start_time = time.time()

if __name__ == '__main__':

    for epoch in range(30):

        scheduler.step()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    end_time = time.time()
    print('Training time: %.2f seconds' % (end_time - start_time))
    print('End Training')

    print('Start Testing')

    correct_total = 0
    total = 0
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct_total += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    print('Overall Accuracy: %d %%' %
          (100 * correct_total / total))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i],
               100 * class_correct[i] / class_total[i]))
