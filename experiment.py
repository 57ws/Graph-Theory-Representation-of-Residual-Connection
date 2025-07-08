import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)


# 残差块实现
class VanishingResidualBlock(nn.Module):
    def __init__(self, in_features, config_type, activation_alpha=0.05):
        super().__init__()
        self.config_type = config_type
        self.activation_alpha = activation_alpha

        # 主干网络的两层线性变换
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(in_features)  # 添加BatchNorm
        self.fc2 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)  # 添加BatchNorm

        # 根据配置类型决定激活函数的使用,使用Tanh模拟更强的梯度消失情景
        if config_type == 'control':
            self.activation = nn.ReLU()
            self.residual_activation = None
        elif config_type == 'exp1':
            self.activation = None
            self.residual_activation = nn.ReLU()
        elif config_type == 'exp2':
            self.activation = nn.ReLU()
            self.residual_activation = nn.ReLU()

    def forward(self, x):
        identity = x

        # 主干网络第一层
        out = self.fc1(x)
        out = self.bn1(out)  # BatchNorm

        if self.activation is not None:
            out = self.activation(out+identity)
        else:
            out = out+self.residual_activation(identity)

        identity=out

        # 主干网络第二层
        out = self.fc2(out)
        out = self.bn2(out)  # BatchNorm

        # 残差连接处理
        if self.config_type == 'control':
            out += identity
            out = self.activation(out)
        elif self.config_type == 'exp1' or self.config_type == 'exp2':
            if self.residual_activation is not None:
                identity = self.residual_activation(identity)
            out += identity

        return out


# 深层网络模型
class DeepResNet(nn.Module):
    def __init__(self, num_blocks, hidden_size, num_classes, config_type, activation_alpha=0.05):
        super().__init__()
        self.config_type = config_type
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size

        # 输入层
        self.input_fc = nn.Linear(32 * 32 * 3, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)  # 输入BatchNorm
        self.input_activation = nn.ReLU()

        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            VanishingResidualBlock(hidden_size, config_type, activation_alpha)
            for _ in range(num_blocks)
        ])

        # 输出层
        self.output_fc = nn.Linear(hidden_size, num_classes)

        # 梯度监控
        self.gradient_norms = []
        self.gradient_points = []  # 记录梯度时的层索引

    def forward(self, x):
        # 扁平化输入
        x = x.view(x.size(0), -1)
        x = self.input_fc(x)
        x = self.input_bn(x)  # 输入BatchNorm
        x = self.input_activation(x)

        # 通过所有残差块
        for i, block in enumerate(self.res_blocks):
            x = block(x)

            # 监控梯度：每层都记录
            if self.training:
                # 保留梯度用于监控
                x.retain_grad()
                self.gradient_points.append(x)

        # 输出层
        x = self.output_fc(x)
        return x

    def get_gradient_norms(self):
        """获取并重置梯度范数记录"""
        norms = []
        for tensor in self.gradient_points:
            if tensor.grad is not None:
                norms.append(tensor.grad.norm(2).item())

        # 重置记录
        self.gradient_points = []
        return norms

    def count_dead_neurons(self, device, threshold=1e-2):
        """统计坏死神经元比例 - 使用更合理的阈值"""
        dead_counts = []
        total_neurons = self.hidden_size

        # 检查每个残差块的输出
        with torch.no_grad():
            # 创建随机输入 - 确保在正确设备上
            test_input = torch.randn(64, 32 * 32 * 3, device=device)
            x = self.input_fc(test_input)
            x = self.input_bn(x)
            x = self.input_activation(x)

            for block in self.res_blocks:
                x = block(x)
                # 统计坏死神经元（输出接近零）
                dead_neurons = (x.abs() < threshold).all(dim=0)  # 所有样本上该神经元都接近零
                dead_ratio = dead_neurons.float().mean().item()
                dead_counts.append(dead_ratio)

        return dead_counts


# 实验配置
class ExperimentConfig:
    def __init__(self, name, config_type, num_blocks=8, hidden_size=256, lr=0.001):
        self.name = name
        self.config_type = config_type
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.lr = lr


# 训练和评估函数
def train_model(config, train_loader, test_loader, epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {config.name} model on {device}...")

    # 创建模型
    model = DeepResNet(
        num_blocks=config.num_blocks,
        hidden_size=config.hidden_size,
        num_classes=10,
        config_type=config.config_type
    ).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)  # 添加权重衰减
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # 学习率调度器
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    test_accuracies = []
    gradient_history = []
    dead_neuron_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 - 防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # 记录训练损失
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 评估测试集
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        # 更新学习率
        scheduler.step(avg_loss)

        # 记录梯度和坏死神经元
        gradient_history.append(model.get_gradient_norms())
        dead_neuron_history.append(model.count_dead_neurons(device))

        print(f'Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')

    # 最终坏死神经元统计
    final_dead_neurons = dead_neuron_history[-1]

    return {
        'name': config.name,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'gradient_history': gradient_history,
        'dead_neuron_history': dead_neuron_history,
        'final_dead_neurons': final_dead_neurons,
        'model': model
    }


# 可视化结果
def visualize_results(results):
    plt.figure(figsize=(18, 12))

    # 训练损失曲线
    plt.subplot(2, 2, 1)
    for res in results:
        plt.plot(res['train_losses'], label=res['name'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 测试准确率曲线
    plt.subplot(2, 2, 2)
    for res in results:
        plt.plot(res['test_accuracies'], label=res['name'])
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 梯度范数分布（最后一轮）
    plt.subplot(2, 2, 3)
    for res in results:
        if res['gradient_history'] and res['gradient_history'][-1]:
            last_epoch_gradients = res['gradient_history'][-1]
            layer_indices = np.arange(len(last_epoch_gradients))
            plt.plot(layer_indices, last_epoch_gradients, label=res['name'])
    plt.title('Gradient Norm by Layer Depth (Final Epoch)')
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)

    # 坏死神经元比例（最后一轮）
    plt.subplot(2, 2, 4)
    for res in results:
        dead_neurons = res['final_dead_neurons']
        plt.plot(np.arange(len(dead_neurons)), dead_neurons, label=res['name'])
    plt.title('Dead Neuron Ratio by Block (Final Epoch)')
    plt.xlabel('Residual Block Index')
    plt.ylabel('Dead Neuron Ratio')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('residual_experiment_results.png', dpi=300)
    plt.show()


# 主实验函数
def run_experiment():
    # 准备数据集 (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=2)

    # 定义三种实验配置（减少层数，增加宽度）
    configs = [
        ExperimentConfig('Control (Standard Residual)', 'control',
                         num_blocks=50, hidden_size=256, lr=0.001),
        ExperimentConfig('Exp1 (Residual Path Only)', 'exp1',
                         num_blocks=50, hidden_size=256, lr=0.001),
        ExperimentConfig('Exp2 (Half Activation)', 'exp2',
                         num_blocks=50, hidden_size=256, lr=0.001)
    ]

    # 运行所有实验
    results = []
    for config in configs:
        result = train_model(config, train_loader, test_loader, epochs=15)
        results.append(result)

    # 可视化结果
    visualize_results(results)

    # 打印最终坏死神经元统计
    print("\nFinal Dead Neuron Ratios:")
    for res in results:
        avg_dead = np.mean(res['final_dead_neurons'])
        max_dead = np.max(res['final_dead_neurons'])
        print(f"{res['name']}: Average={avg_dead:.4f}, Max={max_dead:.4f}")


# 运行实验
if __name__ == "__main__":
    run_experiment()
