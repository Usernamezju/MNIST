"""
MNIST 模型集成推理
===========================================
- 加载所有可用模型权重
- 概率平均集成（soft voting）
- 输出每个单模型准确率 + 各种集成组合准确率
- 仅用于验证集成上限，不作为部署方案
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from itertools import combinations
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ============================================================
#  网络定义
# ============================================================

# 2层CNN（无BN）
class CNN2(nn.Module):
    def __init__(self, dropout=0.3, fc_hidden=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, fc_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 10))
    def forward(self, x):
        return self.fc(self.conv2(self.conv1(x)))


# 4层CNN（无BN）
class CNN4(nn.Module):
    def __init__(self, dropout=0.5, fc_hidden=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, fc_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 10))
    def forward(self, x):
        return self.fc(self.conv4(self.conv3(self.conv2(self.conv1(x)))))


# 4层CNN（有BN）
class CNN4BN(nn.Module):
    def __init__(self, dropout=0.5, fc_hidden=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, fc_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 10))
    def forward(self, x):
        return self.fc(self.conv4(self.conv3(self.conv2(self.conv1(x)))))


# ============================================================
#  所有可用模型配置
#  name: 显示名称
#  cls:  网络类
#  path: 权重文件路径
#  kwargs: 构造函数参数
# ============================================================
MODEL_CONFIGS = [
    {
        'name': '2层CNN  SGD基线',
        'cls':  CNN2,
        'path': 'mnist_best_model.pth',
        'kwargs': {'dropout': 0.3, 'fc_hidden': 128}
    },
    {
        'name': '2层CNN  Adam微调',
        'cls':  CNN2,
        'path': 'finetune_adam_best.pth',
        'kwargs': {'dropout': 0.3, 'fc_hidden': 128}
    },
    {
        'name': '2层CNN  余弦微调',
        'cls':  CNN2,
        'path': 'finetune_cosine_best.pth',
        'kwargs': {'dropout': 0.3, 'fc_hidden': 128}
    },
    {
        'name': '2层CNN  SGD+余弦 从头',
        'cls':  CNN2,
        'path': 'scratch_cosine_best.pth',
        'kwargs': {'dropout': 0.3, 'fc_hidden': 128}
    },
    {
        'name': '2层CNN  纯Adam 从头',
        'cls':  CNN2,
        'path': 'adam_only_best.pth',
        'kwargs': {'dropout': 0.3, 'fc_hidden': 128}
    },
    {
        'name': '2层CNN  Adam+余弦 从头',
        'cls':  CNN2,
        'path': 'exp237_adam_cosine_best.pth',
        'kwargs': {'dropout': 0.3, 'fc_hidden': 128}
    },
    {
        'name': '4层CNN  无BN Adam+余弦',
        'cls':  CNN4,
        'path': 'deep4_adam_cosine_best.pth',
        'kwargs': {'dropout': 0.5, 'fc_hidden': 256}
    },
    {
        'name': '4层CNN  BN+Hold Adam+余弦',
        'cls':  CNN4BN,
        'path': 'deep4_bn_hold_best.pth',
        'kwargs': {'dropout': 0.5, 'fc_hidden': 256}
    },
]

# ============================================================
#  数据加载
# ============================================================
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set    = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    print(f"测试集: {len(test_set)} 张\n")
    return test_loader


# ============================================================
#  加载单个模型
# ============================================================
def load_model(cfg):
    model = cfg['cls'](**cfg['kwargs']).to(DEVICE)
    if not os.path.exists(cfg['path']):
        print(f"  [跳过] 权重文件不存在: {cfg['path']}")
        return None
    model.load_state_dict(torch.load(cfg['path'], map_location=DEVICE))
    model.eval()
    return model


# ============================================================
#  获取所有模型的 softmax 概率输出
#  返回 shape: (N_samples, 10)
# ============================================================
@torch.no_grad()
def get_probs(model, loader):
    all_probs = []
    for X, _ in loader:
        X = X.to(DEVICE)
        logits = model(X)
        probs  = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)   # (10000, 10)


# ============================================================
#  计算准确率
# ============================================================
def calc_acc(probs, labels):
    preds = probs.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ============================================================
#  主流程
# ============================================================
def main():
    test_loader = load_test_data()

    # 收集所有测试集标签
    all_labels = torch.cat([y for _, y in test_loader])

    # ── 1. 加载所有模型，计算单模型准确率 ──────────────────
    print("=" * 60)
    print("单模型准确率")
    print("=" * 60)

    loaded   = []   # (name, probs)
    for cfg in MODEL_CONFIGS:
        model = load_model(cfg)
        if model is None:
            continue
        probs = get_probs(model, test_loader)
        acc   = calc_acc(probs, all_labels)
        print(f"  {cfg['name']:<30}  acc = {acc:.4f}")
        loaded.append({'name': cfg['name'], 'probs': probs})

    if len(loaded) < 2:
        print("\n可用模型不足 2 个，无法集成。")
        return

    # ── 2. 全量集成 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("集成结果")
    print("=" * 60)

    # 全部模型平均
    all_probs_stack = torch.stack([m['probs'] for m in loaded], dim=0)
    ensemble_probs  = all_probs_stack.mean(dim=0)
    ensemble_acc    = calc_acc(ensemble_probs, all_labels)
    print(f"\n  全部 {len(loaded)} 个模型集成       acc = {ensemble_acc:.4f}")

    # ── 3. 只集成强模型（acc > 0.993）───────────────────────
    strong = [m for m in loaded
              if calc_acc(m['probs'], all_labels) > 0.993]
    if len(strong) >= 2:
        strong_stack = torch.stack([m['probs'] for m in strong], dim=0)
        strong_acc   = calc_acc(strong_stack.mean(dim=0), all_labels)
        names        = ' + '.join([m['name'].strip() for m in strong])
        print(f"  强模型集成 ({len(strong)} 个)          acc = {strong_acc:.4f}")
        print(f"    → {names}")

    # ── 4. 两两组合集成，找最优对 ───────────────────────────
    print(f"\n  两两组合最优:")
    best_pair_acc  = 0.0
    best_pair_name = ''
    for a, b in combinations(loaded, 2):
        pair_probs = (a['probs'] + b['probs']) / 2
        pair_acc   = calc_acc(pair_probs, all_labels)
        if pair_acc > best_pair_acc:
            best_pair_acc  = pair_acc
            best_pair_name = f"{a['name'].strip()} + {b['name'].strip()}"
    print(f"    acc = {best_pair_acc:.4f}  ({best_pair_name})")

    # ── 5. 三三组合集成，找最优三元组 ───────────────────────
    if len(loaded) >= 3:
        print(f"\n  三三组合最优:")
        best_trio_acc  = 0.0
        best_trio_name = ''
        for trio in combinations(loaded, 3):
            trio_probs = torch.stack([m['probs'] for m in trio]).mean(0)
            trio_acc   = calc_acc(trio_probs, all_labels)
            if trio_acc > best_trio_acc:
                best_trio_acc  = trio_acc
                best_trio_name = ' + '.join([m['name'].strip() for m in trio])
        print(f"    acc = {best_trio_acc:.4f}  ({best_trio_name})")

    # ── 6. 汇总 ─────────────────────────────────────────────
    best_single = max(calc_acc(m['probs'], all_labels) for m in loaded)
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"  最强单模型                    : {best_single:.4f}")
    print(f"  最优两两集成                  : {best_pair_acc:.4f}  (+{best_pair_acc-best_single:+.4f})")
    if len(loaded) >= 3:
        print(f"  最优三三集成                  : {best_trio_acc:.4f}  (+{best_trio_acc-best_single:+.4f})")
    print(f"  全量集成 ({len(loaded)} 个模型)          : {ensemble_acc:.4f}  (+{ensemble_acc-best_single:+.4f})")
    print("=" * 60)
    print("\n注：集成仅用于验证上限，推理成本 ×N，不适合算力稀缺场景。")


if __name__ == '__main__':
    main()