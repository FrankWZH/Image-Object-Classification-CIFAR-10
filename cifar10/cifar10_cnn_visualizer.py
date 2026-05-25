import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 显示图像的函数
def imgshow(img, title=None, predictions=None):
    """
    显示单张图像
    Args:
        img: 输入图像张量
        title: 图像标题
        predictions: 预测结果（可选）
    """
    # 反归一化：img * 0.5 + 0.5
    img = img / 2 + 0.5

    # 如果是torch.Tensor，转为numpy
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    # 调整维度顺序：CHW -> HWC
    if img.shape[0] <= 3:  # 如果是CHW格式
        img = np.transpose(img, (1, 2, 0))

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

    if title:
        plt.title(title, fontsize=14, pad=20)

    # 如果提供了预测结果，在图像下方显示
    if predictions:
        plt.figtext(0.5, 0.05, predictions,
                    ha='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.show()


# 模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def visualize_predictions(correct_examples, incorrect_examples, num_to_show=2):
    """
    可视化正确和错误的预测结果
    Args:
        correct_examples: 正确预测的示例列表
        incorrect_examples: 错误预测的示例列表
        num_to_show: 每种类型显示的图片数量
    """
    print("\n" + "=" * 60)
    print("Visualizing Predictions")
    print("=" * 60)

    # 显示正确预测的图片
    print("\n✅ Correctly Classified Examples:")
    for i, example in enumerate(correct_examples[:num_to_show]):
        image, true_label, pred_label, confidence, all_probs = example

        # 获取top-3预测
        top_k = 3
        top_probs, top_indices = torch.topk(all_probs, top_k)

        # 构建预测信息文本
        info = f"True: {classes[true_label]} ({true_label})\n"
        info += f"Predictedm: {classes[pred_label]} ({pred_label})\n"
        info += f"Confidence: {confidence:.1%}\n\n"
        info += "Top 3 predictions:\n"

        for j in range(top_k):
            class_idx = top_indices[j].item()
            prob = top_probs[j].item()
            indicator = "→ " if class_idx == pred_label else "  "
            info += f"{indicator}{classes[class_idx]:>10s}: {prob:.1%}\n"

        title = f"Correct Example {i + 1}"
        imgshow(image, title=title, predictions=info)

    # 显示错误预测的图片
    print("\n❌ Incorrectly Classified Examples:")
    for i, example in enumerate(incorrect_examples[:num_to_show]):
        image, true_label, pred_label, confidence, all_probs = example

        # 获取top-3预测
        top_k = 3
        top_probs, top_indices = torch.topk(all_probs, top_k)

        # 构建预测信息文本
        info = f"True: {classes[true_label]} ({true_label})\n"
        info += f"Predicted: {classes[pred_label]} ({pred_label}) ❌\n"
        info += f"Confidence: {confidence:.1%}\n\n"
        info += "Top 3 predictions:\n"

        for j in range(top_k):
            class_idx = top_indices[j].item()
            prob = top_probs[j].item()
            if class_idx == true_label:
                indicator = "✓ "
            elif class_idx == pred_label:
                indicator = "→ "
            else:
                indicator = "  "
            info += f"{indicator}{classes[class_idx]:>10s}: {prob:.1%}\n"

        title = f"Incorrect Example {i + 1}"
        imgshow(image, title=title, predictions=info)


def train_model():
    """训练模型"""
    print("Training the model...")

    # 设置设备
    use_cuda = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() & use_cuda else "cpu")

    # 创建模型
    net = Net()
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练
    start_time = time.time()
    for epoch in range(3):
        training_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, training_loss / 2000))
                training_loss = 0.0

    end_time = time.time()
    print(f'\nTraining completed in {end_time - start_time:.2f} seconds')

    return net, device


def test_and_collect_examples(net, device):
    """测试模型并收集正确和错误的预测示例"""
    print("\nTesting the model and collecting examples...")

    # 存储正确和错误的预测示例
    correct_examples = []
    incorrect_examples = []

    net.eval()  # 设置为评估模式

    correct_total = 0
    total = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            # 计算softmax概率
            probabilities = F.softmax(outputs, dim=1)

            total += labels.size(0)
            correct_total += (predicted == labels).sum().item()

            # 收集每个样本的预测信息
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confidence = probabilities[i][pred_label].item()
                all_probs = probabilities[i]

                # 获取原始图像（反归一化后）
                original_image = images[i].cpu()

                example = (original_image, true_label, pred_label, confidence, all_probs)

                if pred_label == true_label:
                    correct_examples.append(example)
                else:
                    incorrect_examples.append(example)

    accuracy = 100 * correct_total / total
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    print(f'Total correct predictions: {len(correct_examples)}')
    print(f'Total incorrect predictions: {len(incorrect_examples)}')

    return correct_examples, incorrect_examples, accuracy


def analyze_predictions(correct_examples, incorrect_examples):
    """分析预测结果"""
    print("\n" + "=" * 60)
    print("Prediction Analysis")
    print("=" * 60)

    # 分析正确预测的置信度分布
    if correct_examples:
        correct_confidences = [ex[3] for ex in correct_examples]
        avg_correct_confidence = np.mean(correct_confidences)
        min_correct_confidence = np.min(correct_confidences)
        max_correct_confidence = np.max(correct_confidences)

        print(f"\n✅ Correct Predictions ({len(correct_examples)} examples):")
        print(f"   Average confidence: {avg_correct_confidence:.1%}")
        print(f"   Min confidence: {min_correct_confidence:.1%}")
        print(f"   Max confidence: {max_correct_confidence:.1%}")

    # 分析错误预测的置信度分布
    if incorrect_examples:
        incorrect_confidences = [ex[3] for ex in incorrect_examples]
        avg_incorrect_confidence = np.mean(incorrect_confidences)
        min_incorrect_confidence = np.min(incorrect_confidences)
        max_incorrect_confidence = np.max(incorrect_confidences)

        print(f"\n❌ Incorrect Predictions ({len(incorrect_examples)} examples):")
        print(f"   Average confidence: {avg_incorrect_confidence:.1%}")
        print(f"   Min confidence: {min_incorrect_confidence:.1%}")
        print(f"   Max confidence: {max_incorrect_confidence:.1%}")

    # 找出最高置信度的错误预测
    if incorrect_examples:
        # 按置信度排序（降序）
        incorrect_examples_sorted = sorted(incorrect_examples,
                                           key=lambda x: x[3],
                                           reverse=True)

        print("\n🔝 Top 3 most confident but incorrect predictions:")
        for i in range(min(3, len(incorrect_examples_sorted))):
            image, true_label, pred_label, confidence, all_probs = incorrect_examples_sorted[i]
            print(f"   {i + 1}. True: {classes[true_label]}, "
                  f"Predicted: {classes[pred_label]}, "
                  f"Confidence: {confidence:.1%}")


def main():
    """主函数"""
    print("CIFAR-10 Classification with CNN")
    print("=" * 50)

    # 1. 训练模型
    net, device = train_model()

    # 2. 测试并收集示例
    correct_examples, incorrect_examples, accuracy = test_and_collect_examples(net, device)

    # 3. 分析预测结果
    analyze_predictions(correct_examples, incorrect_examples)

    # 4. 可视化示例
    if correct_examples and incorrect_examples:
        visualize_predictions(correct_examples, incorrect_examples, num_to_show=2)
    else:
        print("\nNo examples collected for visualization.")

    # 5. 显示汇总信息
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total test images: {len(testset)}")
    print(f"Model accuracy: {accuracy:.2f}%")
    print(f"Correct predictions available for display: {len(correct_examples)}")
    print(f"Incorrect predictions available for display: {len(incorrect_examples)}")

    return net, correct_examples, incorrect_examples


if __name__ == '__main__':
    # 运行主程序
    net, correct_examples, incorrect_examples = main()

    # 如果想要单独测试显示更多图片，可以调用这个函数
    # visualize_predictions(correct_examples, incorrect_examples, num_to_show=5)