import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ============================================================
#  超参数配置区（调参从这里入手！）
# ============================================================
LEARNING_RATE = 0.1     # 试试: 0.001 / 0.01 / 0.1
BATCH_SIZE = 64     # 试试: 32 / 64 / 256 / 1024
EPOCHS = 40     # 轮数
HIDDEN1 = 256      # 第一隐藏层神经元数
HIDDEN2 = 128      # 第二隐藏层神经元数
WEIGHT_DECAY = 0     # L2 正则化系数，0 表示关闭
MOMENTUM = 0
LABEL_SMOOTHING = 0    # 试试 0.0（关闭） / 0.05 / 0.1 / 0.2
DROPOUT_RATE = 0.3
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
def forward(X, params, is_training=True):
    """
    is_training: 只有训练时才应用 Dropout
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    # 第一层
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    
    # --- Dropout 1 ---
    M1 = None
    if is_training and DROPOUT_RATE > 0:
        # 生成 0/1 掩码，存活概率为 1 - DROPOUT_RATE
        M1 = (np.random.rand(*A1.shape) > DROPOUT_RATE).astype(float)
        # Inverted Dropout: 除以 (1-p) 保持期望不变
        A1 = (A1 * M1) / (1.0 - DROPOUT_RATE)

    # 第二层
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    
    # --- Dropout 2 ---
    M2 = None
    if is_training and DROPOUT_RATE > 0:
        M2 = (np.random.rand(*A2.shape) > DROPOUT_RATE).astype(float)
        A2 = (A2 * M2) / (1.0 - DROPOUT_RATE)

    # 输出层 (不加 Dropout)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'M1': M1,
             'Z2': Z2, 'A2': A2, 'M2': M2, 'Z3': Z3, 'A3': A3}
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
    m = len(y)
    X = cache['X']
    Z1, A1, M1 = cache['Z1'], cache['A1'], cache['M1']
    Z2, A2, M2 = cache['Z2'], cache['A2'], cache['M2']
    A3 = cache['A3']

    # 第三层（输出层）
    K = 10
    smooth_labels = np.full((m, K), LABEL_SMOOTHING / (K - 1))
    smooth_labels[np.arange(m), y] = 1.0 - LABEL_SMOOTHING
    dZ3 = (A3 - smooth_labels) / m
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0, keepdims=True)

    # 第二层（隐藏层 2）
    dA2 = dZ3 @ params['W3'].T
    # --- Dropout Backprop ---
    if M2 is not None:
        dA2 = (dA2 * M2) / (1.0 - DROPOUT_RATE)
    
    dZ2 = dA2 * relu_grad(Z2)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)

    # 第一层（隐藏层 1）
    dA1 = dZ2 @ params['W2'].T
    # --- Dropout Backprop ---
    if M1 is not None:
        dA1 = (dA1 * M1) / (1.0 - DROPOUT_RATE)
        
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
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

            A3, cache = forward(Xb, params, is_training=True)
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
    A3, _ = forward(X, params, is_training=False)
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
    print("训练曲线已保存为 training_curve_dropout.png")




# ============================================================
#  入口
# ============================================================
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    params, history = train(X_train, y_train, X_test, y_test)
    plot_history(history)
    print(f"\n最终测试集准确率: {evaluate(X_test, y_test, params):.4f}")
