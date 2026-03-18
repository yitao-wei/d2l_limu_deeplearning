from utils import *

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
    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.normal_(m.weight, std=0.01)

    # Xavier初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)  

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

