"""
MNIST 手写数字识别 - CNN 版本
===========================================
使用 PyTorch 自动求导，你只需要手搓 CNN 网络结构定义部分。
其余训练流程、评估、可视化已经写好。

网络结构参考（你来决定细节）:
  输入 (1×28×28)
  → Conv2d → ReLU → MaxPool
  → Conv2d → ReLU → MaxPool
  → Flatten
  → Linear → ReLU → Dropout
  → Linear (输出10类)

安装依赖: pip install torch torchvision matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
#  超参数配置区
# ============================================================
LEARNING_RATE = 0.001    # Adam 默认 0.001，比 SGD 稳很多
BATCH_SIZE    = 64       # 试试 32 / 128
EPOCHS        = 10
DROPOUT_RATE  = 0.5      # Dropout 丢弃率，试试 0.3 / 0.5
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"使用设备: {DEVICE}")

# ============================================================
#  1. 数据加载
# ============================================================
def load_data():
    # 相比 MLP 版本多了 Normalize，将像素值标准化到均值0方差1
    # 均值 0.1307 和标准差 0.3081 是 MNIST 数据集的统计值
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"训练集: {len(train_set)} 张, 测试集: {len(test_set)} 张")
    return train_loader, test_loader


# ============================================================
#  2. CNN 网络结构定义（你来填！）
# ============================================================
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()

        # TODO: 定义卷积层部分 (self.conv_layers)
        # 提示: 使用 nn.Sequential 把多个层打包在一起
        #
        # 可参考结构:
        #   第一个卷积块:
        #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        #     nn.ReLU()
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        #     → 输出尺寸: (batch, 32, 14, 14)
        #
        #   第二个卷积块:
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        #     nn.ReLU()
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        #     → 输出尺寸: (batch, 64, 7, 7)
        #
        # 关键参数说明:
        #   in_channels:  输入的通道数（第一层是灰度图所以是1，之后等于上一层的out_channels）
        #   out_channels: 卷积核数量，决定提取多少种特征，试试 16/32/64
        #   kernel_size:  卷积核大小，3×3 是最常用的
        #   padding=1:    让卷积后尺寸不变（配合 kernel_size=3 使用）
        #   MaxPool2d(2): 长宽各缩小一半
        self.conv_layers = None  # ← 替换这一行

        # TODO: 定义全连接层部分 (self.fc_layers)
        # 提示:
        #   经过两次 MaxPool(2)，28×28 → 14×14 → 7×7
        #   所以 Flatten 后的维度是 64 × 7 × 7 = 3136
        #
        #   nn.Flatten()
        #   nn.Linear(3136, 128)
        #   nn.ReLU()
        #   nn.Dropout(DROPOUT_RATE)   ← 防过拟合，训练时随机丢弃神经元
        #   nn.Linear(128, 10)         ← 输出10个类别的 logits（不需要加Softmax，损失函数会处理）
        self.fc_layers = None  # ← 替换这一行

    def forward(self, x):
        # TODO: 实现前向传播
        # 提示: 先过卷积层，再过全连接层，两行就够了
        #   x = self.conv_layers(x)
        #   x = self.fc_layers(x)
        #   return x
        pass  # ← 替换这一行


# ============================================================
#  3. 训练函数（已完成，无需修改）
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()       # 清空上一步的梯度
        output = model(X)           # 前向传播
        loss   = criterion(output, y)
        loss.backward()             # 反向传播（PyTorch 自动求导）
        optimizer.step()            # 更新参数

        total_loss += loss.item() * len(y)
        correct    += (output.argmax(dim=1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            loss   = criterion(output, y)

            total_loss += loss.item() * len(y)
            correct    += (output.argmax(dim=1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


# ============================================================
#  4. 主训练循环
# ============================================================
def train(train_loader, test_loader):
    model     = MnistCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()   # 内部已包含 Softmax，所以网络输出不需要再加
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率调度：每5轮学习率乘以0.5，帮助后期精细收敛
    # 试试注释掉这行，对比有无 scheduler 的收敛曲线
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8} | {'LR':>8}")
    print("-" * 65)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
              f"{test_loss:>9.4f} | {test_acc:>8.4f} | {current_lr:>8.6f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

    return model, history


# ============================================================
#  5. 可视化
# ============================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'],  label='Test Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['test_acc'],  label='Test Acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()

    plt.tight_layout()
    plt.savefig('cnn_training_curve.png', dpi=120)
    plt.show()


def visualize_predictions(model, test_loader, n=10):
    model.eval()
    X, y = next(iter(test_loader))
    X, y = X[:n].to(DEVICE), y[:n]

    with torch.no_grad():
        preds = model(X).argmax(dim=1).cpu()

    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i, ax in enumerate(axes):
        img = X[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')
        color = 'green' if preds[i] == y[i] else 'red'
        ax.set_title(f"pred:{preds[i].item()}\ntrue:{y[i].item()}", color=color, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('cnn_predictions.png', dpi=120)
    plt.show()


def visualize_conv_filters(model):
    """可视化第一层卷积核，看看网络学到了什么特征"""
    filters = model.conv_layers[0].weight.data.cpu()  # shape: (out_ch, 1, 3, 3)
    n = filters.shape[0]
    fig, axes = plt.subplots(4, n // 4, figsize=(n // 4 * 1.5, 6))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(filters[i, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle('第一层卷积核可视化')
    plt.tight_layout()
    plt.savefig('conv_filters.png', dpi=120)
    plt.show()


# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    train_loader, test_loader = load_data()
    model, history = train(train_loader, test_loader)
    plot_history(history)
    visualize_predictions(model, test_loader)
    visualize_conv_filters(model)
    print(f"\n最终测试集准确率: {history['test_acc'][-1]:.4f}")