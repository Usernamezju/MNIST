"""
MNIST 手写数字识别 - 从零手搓神经网络
===========================================
网络结构: 全连接网络 784 -> 256 -> 128 -> 10

待你填写的部分用 TODO 标注，并附有提示。
运行环境: pip install numpy matplotlib torchvision (仅用 torchvision 加载数据集)
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ============================================================
#  超参数配置区（调参从这里入手！）
# ============================================================
LEARNING_RATE = 1     # 试试: 0.001 / 0.01 / 0.1
BATCH_SIZE = 64       # 试试: 32 / 64 / 256 / 1024
EPOCHS = 200      # 轮数
HIDDEN1 = 256      # 第一隐藏层神经元数
HIDDEN2 = 128      # 第二隐藏层神经元数
WEIGHT_DECAY = 1e-4     # L2 正则化系数，0 表示关闭
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1    # 试试 0.0（关闭） / 0.05 / 0.1 / 0.2
# ============================================================
#  1. 数据加载与预处理
# ============================================================


def load_data():
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root='./data', train=True,  download=True, transform=transform)
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    def to_numpy(dataset):
        X = dataset.data.numpy().reshape(-1, 784) / 255.0   # 归一化到 [0,1]
        y = dataset.targets.numpy()
        return X, y

    X_train, y_train = to_numpy(train_set)
    X_test,  y_test = to_numpy(test_set)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ============================================================
#  2. 激活函数（补全 TODO 部分）
# ============================================================
def relu(z):
    # TODO: 实现 ReLU 激活函数
    # 提示: max(0, z)，可以用 np.maximum
    return np.maximum(0, z)
    pass


def relu_grad(z):
    # TODO: 实现 ReLU 的导数
    # 提示: z>0 则为1，否则为0，可以用 (z > 0).astype(float)
    return (z > 0).astype(float)
    pass


def softmax(z):
    # TODO: 实现数值稳定的 Softmax
    # 提示: 先减去每行最大值防止溢出
    #   exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    #   return exp_z / exp_z.sum(axis=1, keepdims=True)
    # 可以给老师挖坑，为什么要分子分母同时除以一个数
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)
    pass


# ============================================================
#  3. 权重初始化
# ============================================================
def init_params():
    """
    He 初始化 (适合 ReLU): W ~ N(0, sqrt(2/fan_in))
    也可以试试 Xavier 初始化: W ~ N(0, sqrt(1/fan_in))
    """
    # 可以给老师挖一个坑，为什么要除以784
    params = {
        'W1': np.random.randn(784,    HIDDEN1) * np.sqrt(2.0 / 784),
        'b1': np.zeros((1, HIDDEN1)),
        'W2': np.random.randn(HIDDEN1, HIDDEN2) * np.sqrt(2.0 / HIDDEN1),
        'b2': np.zeros((1, HIDDEN2)),
        'W3': np.random.randn(HIDDEN2, 10) * np.sqrt(2.0 / HIDDEN2),
        'b3': np.zeros((1, 10)),
    }
    return params


# ============================================================
#  4. 前向传播
# ============================================================
def forward(X, params):
    """
    返回 cache 供反向传播使用
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    # TODO: 实现三层前向传播
    # 提示:
    #   Z1 = X  @ W1 + b1         # shape: (batch, HIDDEN1)
    #   A1 = relu(Z1)
    #   Z2 = A1 @ W2 + b2         # shape: (batch, HIDDEN2)
    #   A2 = relu(Z2)
    #   Z3 = A2 @ W3 + b3         # shape: (batch, 10)
    #   A3 = softmax(Z3)           # 输出概率
    Z1, A1, Z2, A2, Z3, A3 = None, None, None, None, None, None
    Z1 = X  @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    cache = {'X': X, 'Z1': Z1, 'A1': A1,
             'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache


# ============================================================
#  5. 损失函数：交叉熵
# ============================================================
# 给老师挖坑：过拟合怎么办？？
def cross_entropy_loss(A3, y, params):
    m = len(y)
    K = 10  # 类别数

    # 构造软标签
    smooth_labels = np.full((m, K), LABEL_SMOOTHING / (K - 1))
    smooth_labels[np.arange(m), y] = 1.0 - LABEL_SMOOTHING

    # 软标签版交叉熵
    loss = -np.sum(smooth_labels * np.log(A3 + 1e-9)) / m
    return loss


# ============================================================
#  6. 反向传播（核心！）
# ============================================================
def backward(cache, y, params):
    """
    返回梯度字典 grads
    """
    m = len(y)
    X = cache['X']
    Z1, A1 = cache['Z1'], cache['A1']
    Z2, A2 = cache['Z2'], cache['A2']
    A3 = cache['A3']

    # TODO: 实现反向传播
    # 提示（从输出层往输入层推导）:
    #
    # 第三层梯度
    K = 10
    smooth_labels = np.full((m, K), LABEL_SMOOTHING / (K - 1))
    smooth_labels[np.arange(m), y] = 1.0 - LABEL_SMOOTHING

    dZ3 = (A3 - smooth_labels) / m  # 替换原来的 dZ3 三行
    dZ3[np.arange(m), y] -= 1
    dZ3 /= m
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0, keepdims=True)
    # 第二层梯度
    dA2 = dZ3 @ params['W3'].T
    dZ2 = dA2 * relu_grad(Z2)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)
    
    # 第一层梯度
    dA1=dZ2@ params['W2'].T
    dZ1=dA1*relu_grad(Z1)
    dW1=X.T@dZ1
    db1=dZ1.sum(axis=0,keepdims=True)
    grads = {'W1': dW1, 'b1': db1,
             'W2': dW2, 'b2': db2,
             'W3': dW3, 'b3': db3}
    return grads


# ============================================================
#  7. 参数更新：SGD（可扩展为 Momentum / Adam）
# ============================================================
# 给老师挖坑，如果一次偶然造成大幅改变怎么办
def update_params(params, grads):
    # TODO: 实现 SGD 参数更新
    # 提示: param = param - LEARNING_RATE * grad
    # 进阶: 试试加动量 (momentum=0.9)
    for key in params:
        params[key] -= LEARNING_RATE * grads[key]
    return params


# ============================================================
#  8. 训练循环
# ============================================================
def train(X_train, y_train, X_test, y_test):
    params = init_params()
    m = len(y_train)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        # 每轮打乱数据
        idx = np.random.permutation(m)
        X_train, y_train = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, m, BATCH_SIZE):
            Xb = X_train[i: i + BATCH_SIZE]
            yb = y_train[i: i + BATCH_SIZE]

            A3, cache = forward(Xb, params)
            loss = cross_entropy_loss(A3, yb, params)
            grads = backward(cache, yb, params)
            params = update_params(params, grads)

            epoch_loss += loss
            num_batches += 1

        # 每轮统计指标
        avg_loss = epoch_loss / num_batches
        # 这个地方或许老师会问
        train_acc = evaluate(X_train[:5000], y_train[:5000], params)  # 用子集加速
        test_acc = evaluate(X_test, y_test, params)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {avg_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f}")

    return params, history


def evaluate(X, y, params):
    A3, _ = forward(X, params)
    preds = np.argmax(A3, axis=1)
    return np.mean(preds == y)


# ============================================================
#  9. 可视化
# ============================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['test_acc'],  label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=120)
    plt.show()
    print("训练曲线已保存为 training_curve.png")


def visualize_predictions(X_test, y_test, params, n=10):
    """随机展示 n 张预测结果"""
    idx = np.random.choice(len(X_test), n, replace=False)
    A3, _ = forward(X_test[idx], params)
    preds = np.argmax(A3, axis=1)

    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[idx[i]].reshape(28, 28), cmap='gray')
        color = 'green' if preds[i] == y_test[idx[i]] else 'red'
        ax.set_title(
            f"pred:{preds[i]}\ntrue:{y_test[idx[i]]}", color=color, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=120)
    plt.show()


# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    params, history = train(X_train, y_train, X_test, y_test)
    plot_history(history)
    visualize_predictions(X_test, y_test, params)
    print(f"\n最终测试集准确率: {evaluate(X_test, y_test, params):.4f}")
