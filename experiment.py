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


# 自定义激活函数：梯度有界且较小的非线性函数
class VanishingActivation(nn.Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x * torch.tanh(self.alpha * x)


# 残差块实现
class VanishingResidualBlock(nn.Module):
    def __init__(self, in_features, config_type, activation_fn, activation_end):
        super().__init__()
        self.config_type = config_type
        self.activation_fn = activation_fn
        self.activation_end = activation_end

        # 主干网络的两层线性变换
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc2 = nn.Linear(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)

        # 根据配置类型决定激活函数的使用
        self.activation = activation_fn
        self.residual_activation = activation_end

    def forward(self, x):
        identity = x

        # 主干网络第一层
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # 主干网络第二层
        out = self.fc2(out)
        out = self.bn2(out)

        # 残差连接处理
        out = self.residual_activation(out + identity)

        return out


# 深层网络模型
class DeepResNet(nn.Module):
    def __init__(self, num_blocks, hidden_size, num_classes, config_type, activation_fn, activation_end):
        super().__init__()
        self.config_type = config_type
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size

        # 输入层
        self.input_fc = nn.Linear(32 * 32 * 3, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        self.input_activation = nn.GELU()

        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            VanishingResidualBlock(hidden_size, config_type, activation_fn, activation_end)
            for _ in range(num_blocks)
        ])

        # 输出层
        self.output_fc = nn.Linear(hidden_size, num_classes)

        # 梯度监控
        self.gradient_norms = []
        self.gradient_points = []

    def forward(self, x):
        # 扁平化输入
        x = x.view(x.size(0), -1)
        x = self.input_fc(x)
        x = self.input_bn(x)
        x = self.input_activation(x)

        # 通过所有残差块
        for i, block in enumerate(self.res_blocks):
            x = block(x)

            # 监控梯度：每层都记录
            if self.training:
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


# 实验配置
class ExperimentConfig:
    def __init__(self, name, config_type, activation_fn, activation_end, num_blocks=5,
                 hidden_size=256, lr=0.001, weight_decay=1e-4, epochs=15):
        self.name = name
        self.config_type = config_type
        self.activation1_type = activation_fn
        self.activation2_type = activation_end
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def get_activation1_fn(self):
        """根据配置返回激活函数实例[relu,,tanh,gelu,vanishing]"""
        if self.activation1_type == 'relu':
            return nn.ReLU()
        elif self.activation1_type == 'tanh':
            return nn.Tanh()
        elif self.activation1_type == 'gelu':
            return nn.GELU()
        elif self.activation1_type == 'vanishing':
            return VanishingActivation(alpha=0.25)
        else:
            raise ValueError(f"未知的激活函数类型: {self.activation1_type}")

    def get_activation2_fn(self):
        """根据配置返回激活函数实例[relu,,tanh,gelu,vanishing]"""
        if self.activation2_type == 'relu':
            return nn.ReLU()
        elif self.activation2_type == 'tanh':
            return nn.Tanh()
        elif self.activation2_type == 'gelu':
            return nn.GELU()
        elif self.activation2_type == 'vanishing':
            return VanishingActivation(alpha=0.25)
        else:
            raise ValueError(f"未知的激活函数类型: {self.activation2_type}")


# 训练和评估函数
def train_model(config, train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"训练 {config.name} 模型 ({config.activation1_type}主干激活函数) ({config.activation2_type}残差激活函数)在 {device} 上...")

    # 创建模型
    model = DeepResNet(
        num_blocks=config.num_blocks,
        hidden_size=config.hidden_size,
        num_classes=10,
        config_type=config.config_type,
        activation_fn=config.get_activation1_fn(),
        activation_end=config.get_activation2_fn()
    ).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    test_accuracies = []
    gradient_history = []  # 存储每轮梯度的统计信息

    # 新增：存储梯度统计信息
    grad_means = []
    grad_stds = []
    grad_vars = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
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

        # 记录梯度
        gradient_norms = model.get_gradient_norms()
        gradient_history.append(gradient_norms)

        # 计算并存储梯度统计信息
        if gradient_norms:
            grad_norms = np.array(gradient_norms)
            grad_mean = np.mean(grad_norms)
            grad_std = np.std(grad_norms)
            grad_var = np.var(grad_norms)

            grad_means.append(grad_mean)
            grad_stds.append(grad_std)
            grad_vars.append(grad_var)
        else:
            grad_means.append(0)
            grad_stds.append(0)
            grad_vars.append(0)

        # 只打印损失和准确率
        print(f'Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%, '
              f'LR={optimizer.param_groups[0]["lr"]:.2e}')

    return {
        'name': config.name,
        'config': config,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'gradient_history': gradient_history,
        'grad_means': grad_means,
        'grad_stds': grad_stds,
        'grad_vars': grad_vars,
        'model': model
    }


def visualize_results(results):
    plt.figure(figsize=(20, 25))

    # 1. 训练损失曲线
    plt.subplot(3, 2, 1)
    for res in results:
        label = f"{res['name']} ({res['config'].activation1_type} {res['config'].activation2_type})"
        plt.plot(res['train_losses'], '-', label=label)  # 修改这里
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 测试准确率曲线
    plt.subplot(3, 2, 2)
    for res in results:
        label = f"{res['name']} ({res['config'].activation1_type} {res['config'].activation2_type})"
        plt.plot(res['test_accuracies'], '-', label=label)  # 修改这里
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 3. 最后一轮梯度范数随层深变化
    plt.subplot(3, 2, 3)
    for res in results:
        if res['gradient_history'] and res['gradient_history'][-1]:
            last_epoch_gradients = res['gradient_history'][-1]
            layer_indices = np.arange(len(last_epoch_gradients))
            label = f"{res['name']} ({res['config'].activation1_type} {res['config'].activation2_type})"
            plt.plot(layer_indices, last_epoch_gradients, '-', label=label)  # 修改这里
    plt.title('Gradient Norm by Layer Depth (Final Epoch)')
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)

    # 4. 每轮平均梯度范数变化
    plt.subplot(3, 2, 4)
    for res in results:
        label = f"{res['name']} ({res['config'].activation1_type} {res['config'].activation2_type})"
        plt.plot(np.arange(len(res['grad_means'])), res['grad_means'], '-', label=label)  # 修改这里
    plt.title('Mean Gradient Norm per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Gradient Norm')
    plt.legend()
    plt.grid(True)

    # 5. 每轮梯度方差变化
    plt.subplot(3, 2, 5)
    for res in results:
        label = f"{res['name']} ({res['config'].activation1_type} {res['config'].activation2_type})"
        plt.plot(np.arange(len(res['grad_vars'])), res['grad_vars'], '-', label=label)  # 修改这里
    plt.title('Gradient Variance per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Variance')
    plt.legend()
    plt.grid(True)

    # 6. 每轮梯度标准差变化
    plt.subplot(3, 2, 6)
    for res in results:
        label = f"{res['name']} ({res['config'].activation1_type} {res['config'].activation2_type})"
        plt.plot(np.arange(len(res['grad_stds'])), res['grad_stds'], '-', label=label)  # 修改这里
    plt.title('Gradient Standard Deviation per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Standard Deviation')
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

    # 定义实验配置
    configs = [
        # 控制组 (标准残差)

        ExperimentConfig(
            name='re',
            config_type='control',
            activation_fn='relu',
            activation_end='relu',
            num_blocks=10,
            hidden_size=256,
            lr=0.001,
            epochs=20
        ),

        ExperimentConfig(
            name='th',
            config_type='control',
            activation_fn='relu',
            activation_end='relu',
            num_blocks=20,
            hidden_size=256,
            lr=0.001,
            epochs=20
        ),
    ]

    # 运行所有实验
    results = []
    for config in configs:
        result = train_model(config, train_loader, test_loader)
        results.append(result)

    # 可视化结果
    visualize_results(results)

    # 打印最终梯度统计
    print("\n最终梯度统计信息:")
    for res in results:
        # 计算最后一个epoch的梯度统计
        last_epoch_gradients = res['gradient_history'][-1] if res['gradient_history'] else []
        if last_epoch_gradients:
            grad_norms = np.array(last_epoch_gradients)
            grad_mean = np.mean(grad_norms)
            grad_std = np.std(grad_norms)
            grad_var = np.var(grad_norms)
            grad_min = np.min(grad_norms)
            grad_max = np.max(grad_norms)
        else:
            grad_mean = grad_std = grad_var = grad_min = grad_max = float('nan')

        print(f"{res['name']} ({res['config'].activation1_type}):")
        print(f"  梯度范数: 均值={grad_mean:.4e}, 标准差={grad_std:.4e}, 方差={grad_var:.4e}, "
              f"最小值={grad_min:.4e}, 最大值={grad_max:.4e}")
        print("-" * 80)


# 运行实验
if __name__ == "__main__":
    run_experiment()
