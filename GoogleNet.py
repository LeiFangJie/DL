import torch
from torch import nn
from torch.nn import functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        # 向图表中添加多个数据点
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
        self.axes[0].clear()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.pause(0.1)  # 短暂暂停以更新图形

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_iter, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for i, (X, y) in enumerate(pbar):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            
            # 更新进度条显示当前损失和准确率
            pbar.set_postfix({'tran_loss': f'{train_l:.4f}', 'train_acc': f'{train_acc:.4f}'})
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        
        # 每个epoch结束后打印详细信息
        print(f'Epoch {epoch+1}/{num_epochs}: loss {train_l:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}')
    
    print(f'Final loss {train_l:.4f}, final train acc {train_acc:.4f}, '
          f'final test acc {test_acc:.4f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = [0, 0]  # 使用普通列表而不是ModuleList
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            correct = torch.sum(torch.eq(torch.argmax(net(X), dim=1), y))
            metric[0] += correct.item()
            metric[1] += y.numel()
    return metric[0] / metric[1]
# 计算数据集的均值和标准差
def get_dataset_stats(dataset):
    """计算数据集的均值和标准差"""
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    total_samples = 0
    
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    return mean.item(), std.item()

# 首先创建临时数据集计算统计信息
temp_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, 
                                    transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
mean, std = get_dataset_stats(temp_dataset)
print(f"数据集均值: {mean:.4f}, 标准差: {std:.4f}")

batch_size = 128
transform = transforms.Compose([transforms.Resize(224),
                              transforms.ToTensor(),
                              transforms.Normalize((mean,), (std,))])

trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_iter = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_iter = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)

#inception块
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

#模型
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),#
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),#
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),#
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#
#2个inception块+3X3pooling
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),#256_out_channels
                   Inception(256, 128, (128, 192), (32, 96), 64),#480_out_channels
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#480_out_channels

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),#512_out_channels
                   Inception(512, 160, (112, 224), (24, 64), 64),#512_out_channels
                   Inception(512, 128, (128, 256), (24, 64), 64),#512_out_channels
                   Inception(512, 112, (144, 288), (32, 64), 64),#528_out_channels
                   Inception(528, 256, (160, 320), (32, 128), 128),#832_out_channels
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))#832_out_channels

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),#832_out_channels
                   Inception(832, 384, (192, 384), (48, 128), 128),#1024_out_channels
                   nn.AdaptiveAvgPool2d((1,1)),#1024_out_channels,自适应平均池化
                   nn.Flatten())#1024

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs = 0.01, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device('cuda:0'))

