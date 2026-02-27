"""
MNIST 手写数字识别 - MLP + CNN 对比版本
===========================================
网络结构可通过 USE_CNN 切换:
  MLP: 784 -> 256 -> 128 -> 10  (手搓)
  CNN: Conv -> Conv -> FC -> 10 (调库)

运行环境: pip install torch torchvision numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ============================================================
#  模式切换
# ============================================================
USE_CNN = True   # True = CNN模式，False = 原来的手搓MLP模式

# ============================================================
#  超参数配置区
# ============================================================
LEARNING_RATE   = 0.01
LR_DECAY        = 0.95       # 每轮衰减系数
LR_MIN          = 1e-4       # 学习率下限
BATCH_SIZE      = 64
EPOCHS          = 20
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY    = 1e-4

# --- MLP 专用 ---
HIDDEN1  = 256
HIDDEN2  = 128
MOMENTUM = 0.9

# --- CNN 专用（调参重点在这里！）---
CNN_FILTERS_1  = 32      # 第一层卷积核数量，试试 16 / 32 / 64
CNN_FILTERS_2  = 64      # 第二层卷积核数量，试试 32 / 64 / 128
CNN_KERNEL_SIZE = 3      # 卷积核大小，试试 3 / 5
CNN_FC_HIDDEN  = 128     # 全连接层大小，试试 64 / 128 / 256
CNN_DROPOUT    = 0.5     # Dropout 比率，试试 0.3 / 0.5
CNN_USE_BN     = True    # 是否使用 Batch Normalization，True/False 对比效果

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ============================================================
#  1. 数据加载
# ============================================================
def load_data():
    if USE_CNN:
        # CNN 需要保留图像的二维结构，并做标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
        ])
        train_set = datasets.MNIST('./data', train=True,  download=True, transform=transform)
        test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"[CNN模式] 训练集: {len(train_set)} 张, 测试集: {len(test_set)} 张")
        return train_loader, test_loader

    else:
        # MLP 模式：拉平成向量
        transform = transforms.ToTensor()
        train_set = datasets.MNIST('./data', train=True,  download=True, transform=transform)
        test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

        def to_numpy(dataset):
            X = dataset.data.numpy().reshape(-1, 784) / 255.0
            y = dataset.targets.numpy()
            return X, y

        X_train, y_train = to_numpy(train_set)
        X_test,  y_test  = to_numpy(test_set)
        print(f"[MLP模式] 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, y_train, X_test, y_test


# ============================================================
#  2. CNN 网络定义（调库实现）
# ============================================================
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积块1: 输入 (1,28,28) → 输出 (CNN_FILTERS_1, 14, 14)
        layers1 = [
            nn.Conv2d(1, CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE, padding=CNN_KERNEL_SIZE//2),
            nn.ReLU(),
        ]
        if CNN_USE_BN:
            layers1.append(nn.BatchNorm2d(CNN_FILTERS_1))  # BN 放在激活后
        layers1.append(nn.MaxPool2d(2))
        self.conv1 = nn.Sequential(*layers1)

        # 卷积块2: 输入 (CNN_FILTERS_1,14,14) → 输出 (CNN_FILTERS_2, 7, 7)
        layers2 = [
            nn.Conv2d(CNN_FILTERS_1, CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE, padding=CNN_KERNEL_SIZE//2),
            nn.ReLU(),
        ]
        if CNN_USE_BN:
            layers2.append(nn.BatchNorm2d(CNN_FILTERS_2))
        layers2.append(nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(*layers2)

        # 全连接层: Flatten → 隐藏层 → 输出
        fc_input = CNN_FILTERS_2 * 7 * 7   # 两次 MaxPool(2) 后 28→7
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input, CNN_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(CNN_FC_HIDDEN, 10)    # 输出 logits，不加 Softmax
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


# ============================================================
#  3. MLP 部分（手搓，保持原样）
# ============================================================
def relu(z):       return np.maximum(0, z)
def relu_grad(z):  return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def init_params():
    return {
        'W1': np.random.randn(784,    HIDDEN1) * np.sqrt(2.0 / 784),
        'b1': np.zeros((1, HIDDEN1)),
        'W2': np.random.randn(HIDDEN1, HIDDEN2) * np.sqrt(2.0 / HIDDEN1),
        'b2': np.zeros((1, HIDDEN2)),
        'W3': np.random.randn(HIDDEN2, 10)      * np.sqrt(2.0 / HIDDEN2),
        'b3': np.zeros((1, 10)),
    }

def forward_mlp(X, params):
    Z1 = X @ params['W1'] + params['b1'];  A1 = relu(Z1)
    Z2 = A1 @ params['W2'] + params['b2']; A2 = relu(Z2)
    Z3 = A2 @ params['W3'] + params['b3']; A3 = softmax(Z3)
    return A3, {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}

def mlp_loss(A3, y):
    m, K = len(y), 10
    sl = np.full((m, K), LABEL_SMOOTHING / (K - 1))
    sl[np.arange(m), y] = 1.0 - LABEL_SMOOTHING
    return -np.sum(sl * np.log(A3 + 1e-9)) / m

def backward_mlp(cache, y, params):
    m, K = len(y), 10
    A3, A2, A1 = cache['A3'], cache['A2'], cache['A1']
    Z2, Z1, X  = cache['Z2'], cache['Z1'], cache['X']
    sl = np.full((m, K), LABEL_SMOOTHING / (K - 1))
    sl[np.arange(m), y] = 1.0 - LABEL_SMOOTHING
    dZ3 = (A3 - sl) / m
    dW3 = A2.T @ dZ3;              db3 = dZ3.sum(0, keepdims=True)
    dZ2 = (dZ3 @ params['W3'].T) * relu_grad(Z2)
    dW2 = A1.T @ dZ2;              db2 = dZ2.sum(0, keepdims=True)
    dZ1 = (dZ2 @ params['W2'].T) * relu_grad(Z1)
    dW1 = X.T  @ dZ1;              db1 = dZ1.sum(0, keepdims=True)
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

def evaluate_mlp(X, y, params):
    A3, _ = forward_mlp(X, params)
    return np.mean(np.argmax(A3, axis=1) == y)


# ============================================================
#  4. CNN 训练 / 评估
# ============================================================
def train_cnn(train_loader, test_loader):
    model     = MnistCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=LR_DECAY)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CNN 参数量: {total_params:,}")
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
        test_acc   = evaluate_cnn(model, test_loader)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} | {train_loss:>9.4f} | {train_acc:>8.4f} | {test_acc:>7.4f} | {lr:>8.6f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    return model, history

def evaluate_cnn(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            correct += (model(X).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total


# ============================================================
#  5. MLP 训练
# ============================================================
def train_mlp(X_train, y_train, X_test, y_test):
    params   = init_params()
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    m        = len(y_train)
    lr       = LEARNING_RATE
    history  = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    print(f"{'Epoch':>6} | {'Loss':>8} | {'TrainAcc':>8} | {'TestAcc':>7} | {'LR':>8}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        idx = np.random.permutation(m)
        X_train, y_train = X_train[idx], y_train[idx]
        epoch_loss, num_batches = 0.0, 0

        for i in range(0, m, BATCH_SIZE):
            Xb, yb   = X_train[i:i+BATCH_SIZE], y_train[i:i+BATCH_SIZE]
            A3, cache = forward_mlp(Xb, params)
            loss      = mlp_loss(A3, yb)
            grads     = backward_mlp(cache, yb, params)

            # Momentum SGD
            for key in params:
                velocity[key] = MOMENTUM * velocity[key] - lr * grads[key]
                params[key]  += velocity[key]

            epoch_loss  += loss
            num_batches += 1

        lr = max(lr * LR_DECAY, LR_MIN)  # 学习率衰减
        avg_loss  = epoch_loss / num_batches
        train_acc = evaluate_mlp(X_train[:5000], y_train[:5000], params)
        test_acc  = evaluate_mlp(X_test, y_test, params)

        print(f"{epoch:>6} | {avg_loss:>8.4f} | {train_acc:>8.4f} | {test_acc:>7.4f} | {lr:>8.6f}")
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    return params, history


# ============================================================
#  6. 可视化
# ============================================================
def plot_history(history, title=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['test_acc'],  label='Test Acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    fname = 'cnn_curve.png' if USE_CNN else 'mlp_curve.png'
    plt.savefig(fname, dpi=120)
    plt.show()
    print(f"曲线已保存为 {fname}")


# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    if USE_CNN:
        train_loader, test_loader = load_data()
        model, history = train_cnn(train_loader, test_loader)
        plot_history(history, title='CNN')
        print(f"\nCNN 最终测试集准确率: {history['test_acc'][-1]:.4f}")
    else:
        X_train, y_train, X_test, y_test = load_data()
        params, history = train_mlp(X_train, y_train, X_test, y_test)
        plot_history(history, title='MLP')
        print(f"\nMLP 最终测试集准确率: {history['test_acc'][-1]:.4f}")