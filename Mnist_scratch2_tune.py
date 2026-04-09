import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# 超参数
# ============================================================
FINETUNE_LR     = 1e-3
FINETUNE_EPOCHS = 10
BATCH_SIZE      = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前运行设备: {DEVICE}")

# ============================================================
# 模型结构（必须和原训练脚本完全一致）
# ============================================================
CNN_FILTERS_1 = 16
CNN_FILTERS_2 = 32
CNN_FC_HIDDEN = 128
CNN_DROPOUT   = 0.3

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, CNN_FILTERS_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(CNN_FILTERS_1, CNN_FILTERS_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_FILTERS_2 * 7 * 7, CNN_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(CNN_FC_HIDDEN, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc(x)

# ============================================================
# 数据加载
# ============================================================
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_loader = DataLoader(
    datasets.MNIST('./data', train=True,  download=True, transform=train_transform),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=test_transform),
    batch_size=BATCH_SIZE, shuffle=False
)

# ============================================================
# 加载模型 + 冻结卷积层
# ============================================================
model = MnistCNN().to(DEVICE)
model.load_state_dict(torch.load('mnist_best_model.pth', map_location=DEVICE))
print("✅ 原模型权重加载成功\n")

for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.conv2.parameters():
    param.requires_grad = False

print("【参数状态检查】")
for name, param in model.named_parameters():
    status = "训练" if param.requires_grad else "冻结"
    print(f"  {name:<35} [{status}]")

# ============================================================
# 优化器：只传入 FC 层参数
# ============================================================
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=FINETUNE_LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
criterion = nn.CrossEntropyLoss()

# ============================================================
# 微调训练循环
# ============================================================
best_acc = 0.0
print(f"\n{'='*68}")
print(f"{'Epoch':>5} | {'TrainLoss':>9} | {'TrainAcc':>8} | {'TestAcc':>7} | {'LR':>10} |")
print(f"{'='*68}")

for epoch in range(1, FINETUNE_EPOCHS + 1):
    # 训练
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (outputs.argmax(1) == y).sum().item()
        total      += len(y)
    train_loss = total_loss / total
    train_acc  = correct / total

    # 评估
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            test_correct += (model(X).argmax(1) == y).sum().item()
            test_total   += len(y)
    test_acc = test_correct / test_total

    curr_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    # 突破历史最高则保存
    is_best = test_acc > best_acc
    if is_best:
        best_acc = test_acc
        torch.save(model.state_dict(), 'mnist_finetuned_model.pth')

    flag = "  ★ BEST  已保存" if is_best else ""
    print(f"{epoch:>5} | {train_loss:>9.4f} | {train_acc:>8.4f} | {test_acc:>7.4f} | {curr_lr:>10.8f} |{flag}")

print(f"{'='*68}")
print(f"微调结束！最优 Test Acc = {best_acc:.4f}，模型已保存至 mnist_finetuned_model.pth")
