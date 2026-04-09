"""
MNIST 手写数字识别 - CNN 版本
===========================================
CNN 特征图尺寸变化:
  输入        (1,  28, 28)
  Conv1+Pool  (32, 14, 14)
  Conv2+Pool  (64,  7,  7)
  Conv3+Pool  (10,  4,  4)
  GlobalAvgPool → (10,)  直接输出logits

运行环境: pip install torch torchvision matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
#  超参数配置区
# ============================================================
LEARNING_RATE   = 0.001
LR_DECAY        = 0.95
BATCH_SIZE      = 64
EPOCHS          = 20
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY    = 1e-4
CNN_FILTERS_1   = 16
CNN_FILTERS_2   = 32
CNN_KERNEL_SIZE = 3
CNN_DROPOUT     = 0.3
CNN_USE_BN      = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ============================================================
#  1. 数据加载
# ============================================================
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set    = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_set     = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练集: {len(train_set)} 张, 测试集: {len(test_set)} 张")
    return train_loader, test_loader


# ============================================================
#  2. 网络定义
# ============================================================
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积块1: (1, 28, 28) → (32, 14, 14)
        layers1 = [nn.Conv2d(1, CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE, padding=CNN_KERNEL_SIZE//2), nn.ReLU()]
        if CNN_USE_BN: layers1.append(nn.BatchNorm2d(CNN_FILTERS_1))
        layers1.append(nn.MaxPool2d(2))
        self.conv1 = nn.Sequential(*layers1)

        # 卷积块2: (32, 14, 14) → (64, 7, 7)
        layers2 = [nn.Conv2d(CNN_FILTERS_1, CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE, padding=CNN_KERNEL_SIZE//2), nn.ReLU()]
        if CNN_USE_BN: layers2.append(nn.BatchNorm2d(CNN_FILTERS_2))
        layers2.append(nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(*layers2)

        # 卷积块3: (64, 7, 7) → (10, 4, 4)，10个核对应10个类别
        layers3 = [nn.Conv2d(CNN_FILTERS_2, 10, kernel_size=CNN_KERNEL_SIZE, padding=1), nn.ReLU()]
        if CNN_USE_BN: layers3.append(nn.BatchNorm2d(10))
        layers3.append(nn.MaxPool2d(2, ceil_mode=True))  # ceil_mode: 7→4 而非 7→3
        self.conv3 = nn.Sequential(*layers3)

        # GlobalAvgPool: (10, 4, 4) → (10,)，无线性层
        self.gap = nn.Sequential(
            nn.Dropout(CNN_DROPOUT),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return x


# ============================================================
#  3. 训练
# ============================================================
def train(train_loader, test_loader):
    model     = MnistCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=LR_DECAY)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Epoch':>6} | {'TrainLoss':>9} | {'TrainAcc':>8} | {'TestAcc':>7} | {'LR':>8}")
    print("-" * 55)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)

        train_loss = total_loss / total
        train_acc  = correct / total
        test_acc   = evaluate(model, test_loader)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} | {train_loss:>9.4f} | {train_acc:>8.4f} | {test_acc:>7.4f} | {lr:>8.6f}")
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    return model, history


# ============================================================
#  4. 评估
# ============================================================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            correct += (model(X).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total


# ============================================================
#  5. 可视化
# ============================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['test_acc'],  label='Test Acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig('cnn_curve.png', dpi=120)
    plt.show()


# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    train_loader, test_loader = load_data()
    model, history = train(train_loader, test_loader)
    plot_history(history)
    print(f"\n最终测试集准确率: {history['test_acc'][-1]:.4f}")