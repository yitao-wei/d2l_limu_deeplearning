import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib

matplotlib.use('TkAgg')  #设置稳定的后端
import matplotlib.pyplot as plt
from d2l import torch as d2l


# ===== 所有辅助函数定义 =====
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater, device):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


#  稳定版 Animator（Windows 兼容）
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []

        # 只启用一次交互模式
        plt.ion()

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        self.config_axes = lambda: self._set_axes(self.axes[0], xlabel, ylabel,
                                                  xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.config_axes()

        #  初始显示一次窗口
        self.fig.show()

    def _set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置坐标轴（简化版）"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        if xlim:
            axes.set_xlim(xlim)
        if ylim:
            axes.set_ylim(ylim)
        axes.legend(legend)
        axes.grid(True, alpha=0.3)

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
                #  清除旧线，画新线（避免重叠）
                self.axes[0].cla()
                # 重新设置坐标轴
                self.config_axes()
                # 画所有曲线
                for j in range(len(self.X)):
                    if self.X[j] and self.Y[j]:
                        self.axes[0].plot(self.X[j], self.Y[j], self.fmts[j % len(self.fmts)])

        #  刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #  暂停一小会儿（关键！让窗口有时间刷新）
        plt.pause(0.01)

    def close(self):
        """关闭窗口"""
        plt.close(self.fig)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, device):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, device)
        test_acc = evaluate_accuracy(net, test_iter, device)

        #  动态更新图表
        try:
            animator.add(epoch + 1, train_metrics + (test_acc,))
        except:
            # 窗口关闭也不影响训练
            pass

        #  同时打印进度（双重保障）
        print(f"epoch {epoch + 1:2d}, loss {train_metrics[0]:.4f}, "
              f"train_acc {train_metrics[1]:.3f}, test_acc {test_acc:.3f}")

    train_loss, train_acc = train_metrics
    print(f"\n训练完成！")

    # 训练结束后保持窗口
    plt.ioff()  # 关闭交互模式
    plt.show()  # 阻塞显示，直到关闭窗口

    return train_loss, train_acc


# ===== 所有执行代码放在 __main__ 保护下 =====
if __name__ == '__main__':
    # GPU/CPU 检测
    print("=" * 60)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        device = torch.device('cpu')
        print(" GPU 不可用，使用 CPU")
    print(f"使用设备：{device}")
    print("=" * 60)

    # 模型参数
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    # 定义网络
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, 10)
    )
    net = net.to(device)


    # 权重初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)


    net.apply(init_weights)

    # 训练配置
    num_epochs, lr, batch_size = 10, 0.1, 256
    loss = nn.CrossEntropyLoss()

    # 用 PyTorch 原生 DataLoader
    transform = transforms.Compose([transforms.ToTensor()])

    print("\n 正在加载 Fashion-MNIST 数据集...")
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"数据集加载完成！")

    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 开始训练
    print("\n 开始训练...（图表将动态更新）")
    print("=" * 60)

    try:
        train_loss, train_acc = train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer, device)
    except Exception as e:
        print(f"\n⚠ 训练中断：{e}")

    # 最终评估
    test_acc = evaluate_accuracy(net, test_iter, device)
    print("=" * 60)
    print(f" 最终测试准确率：{test_acc:.4f} ({test_acc * 100:.2f}%)")

    # 保存图片到文件
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存为 training_curve.png")

    print("\n 程序执行完成！")
    print("=" * 60)

