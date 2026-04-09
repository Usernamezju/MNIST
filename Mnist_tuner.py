"""
MNIST 实验 — 4层CNN + BN + Cosine Decay with Hold
===========================================
- 4层卷积 + BatchNorm
- 调度策略: 余弦退火到第 T_MAX 轮后固定在 ETA_MIN（不重启）
- 优化器: Adam
- 无数据增强
- Dropout: 0.5
- 从头训练 40 轮
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

# ============================================================
#  超参数
# ============================================================
BATCH_SIZE      = 64
EPOCHS          = 40
LABEL_SMOOTHING = 0
WEIGHT_DECAY    = 1e-4

CNN_FILTERS_1   = 16
CNN_FILTERS_2   = 32
CNN_FILTERS_3   = 64
CNN_FILTERS_4   = 64
CNN_KERNEL_SIZE = 3
CNN_FC_HIDDEN   = 256
CNN_DROPOUT     = 0.5

INIT_LR   = 0.001
ETA_MIN   = 1e-6
T_MAX     = 20          # 余弦退火周期，之后 lr 固定在 ETA_MIN
SAVE_NAME = 'deep4_bn_hold_best.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
print(f"4层CNN + BN  Adam + Cosine Decay with Hold")
print(f"lr: {INIT_LR} →(余弦退火 {T_MAX} 轮)→ {ETA_MIN} →(固定保持)→ epoch {EPOCHS}")

# ============================================================
#  1. 数据加载（无增强）
# ============================================================
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print("数据增强: 无")

    train_set    = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_set     = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练集: {len(train_set)} 张  测试集: {len(test_set)} 张")
    return train_loader, test_loader


# ============================================================
#  2. 网络定义（4层卷积 + BN）
# ============================================================
class Deep4BnCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积块1: (1, 28, 28) → (16, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, CNN_FILTERS_1, kernel_size=CNN_KERNEL_SIZE,
                      padding=CNN_KERNEL_SIZE // 2),
            nn.BatchNorm2d(CNN_FILTERS_1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 卷积块2: (16, 14, 14) → (32, 7, 7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(CNN_FILTERS_1, CNN_FILTERS_2, kernel_size=CNN_KERNEL_SIZE,
                      padding=CNN_KERNEL_SIZE // 2),
            nn.BatchNorm2d(CNN_FILTERS_2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 卷积块3: (32, 7, 7) → (64, 7, 7)  不池化
        self.conv3 = nn.Sequential(
            nn.Conv2d(CNN_FILTERS_2, CNN_FILTERS_3, kernel_size=CNN_KERNEL_SIZE,
                      padding=CNN_KERNEL_SIZE // 2),
            nn.BatchNorm2d(CNN_FILTERS_3),
            nn.ReLU()
        )

        # 卷积块4: (64, 7, 7) → (64, 7, 7)  不池化
        self.conv4 = nn.Sequential(
            nn.Conv2d(CNN_FILTERS_3, CNN_FILTERS_4, kernel_size=CNN_KERNEL_SIZE,
                      padding=CNN_KERNEL_SIZE // 2),
            nn.BatchNorm2d(CNN_FILTERS_4),
            nn.ReLU()
        )

        # FC: 3136 → 256 → 10（不加 BN，避免与 Dropout 冲突）
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_FILTERS_4 * 7 * 7, CNN_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(CNN_FC_HIDDEN, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


# ============================================================
#  3. 自定义调度器：Cosine Decay with Hold
#     Epoch 0 ~ T_MAX-1 : 余弦退火 INIT_LR → ETA_MIN
#     Epoch T_MAX ~ END  : 固定在 ETA_MIN，不重启
# ============================================================
def build_scheduler(optimizer):
    ratio = ETA_MIN / INIT_LR   # 最小倍率

    def cosine_hold_lr(epoch):
        if epoch < T_MAX:
            # 余弦退火阶段：从 1.0 降到 ratio
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / T_MAX))
            return ratio + (1 - ratio) * cosine
        else:
            # Hold 阶段：固定在最小学习率
            return ratio

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_hold_lr)


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
#  5. 训练
# ============================================================
def train(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer)

    # 打印理论 lr 曲线
    print("\n理论 LR 曲线:")
    for e in [0, 5, 10, 15, 19, 20, 25, 30, 39]:
        ratio = ETA_MIN / INIT_LR
        if e < T_MAX:
            cosine = 0.5 * (1 + math.cos(math.pi * e / T_MAX))
            lr = INIT_LR * (ratio + (1 - ratio) * cosine)
        else:
            lr = ETA_MIN
        tag = '← 退火结束，开始 Hold' if e == T_MAX else ''
        print(f"  Epoch {e+1:>2}: {lr:.7f}  {tag}")

    best_acc = 0.0
    history  = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}

    print(f"\n{'Epoch':>6} | {'TrainLoss':>9} | {'TrainAcc':>8} | {'TestAcc':>7} | "
          f"{'LR':>12} | {'Gap':>7} | {'阶段':>8}")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            total      += len(y)

            if (batch_idx + 1) % 300 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  [Epoch {epoch:>2} | Batch {batch_idx+1:>4}/{len(train_loader)}]"
                      f"  LR: {cur_lr:.7f}  Loss: {loss.item():.4f}", flush=True)

        train_loss = total_loss / total
        train_acc  = correct / total
        test_acc   = evaluate(model, test_loader)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        gap    = train_acc - test_acc
        phase  = '余弦退火' if epoch <= T_MAX else 'Hold 固定'
        flag   = '  ⚠' if gap > 0.01 else ''

        print(f"\n{epoch:>6} | {train_loss:>9.4f} | {train_acc:>8.4f} | {test_acc:>7.4f} | "
              f"{cur_lr:>12.7f} | {gap:>7.4f} | {phase}{flag}")
        print("-" * 80)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['lr'].append(cur_lr)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_NAME)
            print(f"  → 保存最优模型 acc={best_acc:.4f}  →  {SAVE_NAME}")

    print(f"\n最优测试集准确率: {best_acc:.4f}")
    return history, best_acc


# ============================================================
#  6. 可视化
# ============================================================
def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('4层CNN + BN — Cosine Decay with Hold')

    axes[0].plot(history['train_loss'], color='steelblue', label='Train Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_acc'], color='steelblue', label='Train Acc')
    axes[1].plot(history['test_acc'],  color='orange',    label='Test Acc')
    axes[1].axvline(x=T_MAX, color='gray', linestyle='--', linewidth=1, label=f'Hold 起点 (E{T_MAX})')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(history['lr'], color='green', marker='o', markersize=2, label='LR')
    axes[2].axvline(x=T_MAX, color='gray', linestyle='--', linewidth=1, label=f'Hold 起点 (E{T_MAX})')
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log')
    axes[2].legend()

    plt.tight_layout()
    fname = 'deep4_bn_hold_curve.png'
    plt.savefig(fname, dpi=120)
    plt.show()
    print(f"曲线已保存为 {fname}")


# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    train_loader, test_loader = load_data()

    model = Deep4BnCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    print("\n网络结构验证:")
    dummy = torch.zeros(1, 1, 28, 28).to(DEVICE)
    for name, layer in [('conv1', model.conv1), ('conv2', model.conv2),
                         ('conv3', model.conv3), ('conv4', model.conv4)]:
        dummy = layer(dummy)
        print(f"  {name} 输出: {tuple(dummy.shape)}")

    history, best_acc = train(model, train_loader, test_loader)
    plot_history(history)

    print("\n========== 实验对比 ==========")
    print(f"4层CNN  无BN  T_MAX=40  Adam+余弦        : 0.9950  (有震荡)")
    print(f"4层CNN  有BN  T_MAX=20  Adam+余弦(重启)  : (震荡下降)")
    print(f"4层CNN  有BN  余弦+Hold Adam             : {best_acc:.4f}  ← 本次")
    print("================================")