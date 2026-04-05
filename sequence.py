import torch  # 导入PyTorch主库，提供张量计算和自动求导
from torch import nn  # 导入神经网络模块，包含层和损失函数
import matplotlib.pyplot as plt  # 导入绘图库，用于可视化数据
from torch.utils.data import DataLoader, TensorDataset  # 数据加载工具

T = 1000  # 定义序列总长度为1000个时间点
time = torch.arange(1, T + 1, dtype=torch.float32)  # 生成1到1000的时间索引，float32类型
# 生成正弦波+高斯噪声：torch.sin(0.01 * time)生成周期为2π/0.01≈628的正弦波
# torch.normal(0, 0.2, (T,))生成均值为0，标准差0.2的随机噪声
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

plt.figure(figsize=(6, 3))  # 创建新图，设置尺寸为宽6英寸、高3英寸
plt.plot(time.numpy(), x.numpy())  # 将张量转为numpy数组并绘制折线图
plt.xlabel('time')  # 设置x轴标签
plt.ylabel('x')  # 设置y轴标签
plt.xlim(1, 1000)  # 设置x轴显示范围
plt.show()  # 显示图像（阻塞运行直到关闭窗口）

tau = 4  # 定义时间窗口大小：用前4个时刻预测下一个时刻
# 创建(T-tau, tau)=(996, 4)的零张量存储特征，每行是一个样本，每列是一个时间步的特征
features = torch.zeros((T - tau, tau))
# 循环4次，每次偏移一个位置，构建滑动窗口特征
# i=0: x[0:996], i=1: x[1:997], i=2: x[2:998], i=3: x[3:999]
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
# 标签是从第5个元素(tau位置)开始的数据，reshape为(996, 1)
# 每个标签是对应特征之后那个时刻的值
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600  # 设置批次大小16，训练样本数600（前600个用于训练）
# 只有前n_train个样本用于训练
# 将特征和标签打包成数据集，方便DataLoader使用
train_dataset = TensorDataset(features[:n_train], labels[:n_train])
# 创建可迭代的数据加载器，每轮打乱顺序，分批返回数据
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化网络权重的函数
def init_weights(m):
    # 如果是全连接层，使用Xavier均匀分布初始化权重
    # Xavier初始化根据输入输出维度调整权重范围，帮助梯度稳定传播
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    # 定义网络结构：输入层4个特征(对应tau=4)，隐藏层10个单元，输出层1个预测值
    net = nn.Sequential(nn.Linear(4, 10),  # 输入层：4个特征→10个隐藏单元
                        nn.ReLU(),  # 激活函数：引入非线性，负数截断为0
                        nn.Linear(10, 1))  # 输出层：10个隐藏单元→1个预测值
    net.apply(init_weights)  # 对网络所有层应用权重初始化
    return net  # 返回构建好的网络

# 评估给定数据集上的损失
def evaluate_loss(net, data_iter, loss):
    total_loss = 0  # 累计损失
    total_samples = 0  # 累计样本数
    with torch.no_grad():  # 禁用梯度计算（评估时不需要反向传播，节省内存）
        for X, y in data_iter:  # 遍历数据批次
            l = loss(net(X), y)  # 前向传播计算预测值，再计算损失
            total_loss += l.sum().item()  # 累加批次损失（.item()转为Python数值）
            total_samples += y.numel()  # 累加样本数（numel()返回元素总数）
    return total_loss / total_samples  # 返回平均损失

# 均方误差损失，reduction='none'返回每个样本的独立损失（不平均）
# MSELoss计算的是预测值与真实值差的平方
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    # Adam优化器，结合了动量和自适应学习率，lr为学习率
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):  # 外层循环：遍历指定轮数
        for X, y in train_iter:  # 内层循环：遍历训练数据的所有批次
            trainer.zero_grad()  # 清空之前累积的梯度（PyTorch默认累积梯度）
            l = loss(net(X), y)  # 前向传播：预测并计算损失
            l.sum().backward()  # 反向传播：计算梯度（.sum()将批次损失合并标量）
            trainer.step()  # 更新参数：根据梯度调整权重
        # 每轮结束打印损失，使用evaluate_loss计算整个训练集的平均损失
        print(f'epoch {epoch + 1}, '
              f'loss: {evaluate_loss(net, train_iter, loss):f}')

net = get_net()  # 实例化网络
train(net, train_iter, loss, 5, 0.01)  # 训练5轮，学习率0.01

# 单步预测：对所有996个样本进行预测（前向传播）
# 网络输出的是基于前tau=4个时间步的下一个时间步预测值
onestep_preds = net(features)
plt.figure(figsize=(6, 3))  # 创建新图
# 绘制原始数据（.detach()分离计算图，避免影响反向传播历史）
plt.plot(time.numpy(), x.detach().numpy(), label='data')
# 绘制预测值，时间轴从tau=4开始，因为预测的是从第5个点开始的值
plt.plot(time[tau:].numpy(), onestep_preds.detach().numpy(), label='1-step preds')
plt.xlabel('time')  # x轴标签
plt.ylabel('x')  # y轴标签
plt.xlim(1, 1000)  # x轴范围
plt.legend()  # 显示图例，标注data和1-step preds两条线
plt.show()  # 显示图像

# 多步预测（自回归预测）：使用模型自己的预测作为下一步的输入
# 这种方法会在时间轴上不断累积误差
multistep_preds = torch.zeros(T)  # 创建长度为T的零张量存储多步预测结果
# 前n_train+tau个位置用真实数据填充（训练数据+初始窗口）
# n_train=600, tau=4, 所以前604个位置用真实值
multistep_preds[: n_train + tau] = x[: n_train + tau]
# 从第605个位置开始，使用自回归方式进行多步预测
# 每次用前tau=4个预测值（或真实值）来预测下一个值
for i in range(n_train + tau, T):
    # 取前tau个值作为输入，reshape为(1, 4)的批次格式
    # i-tau:i 是从i-4到i-1的切片，包含4个元素
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

# 绘制三合一对比图：原始数据、单步预测、多步预测
plt.figure(figsize=(6, 3))  # 创建新图
# 绘制原始数据（全部1000个点）
plt.plot(time.numpy(), x.detach().numpy(), label='data')
# 绘制单步预测（从tau=4开始，共996个点）
plt.plot(time[tau:].numpy(), onestep_preds.detach().numpy(), label='1-step preds')
# 绘制多步预测（只绘制预测部分，从n_train+tau=604开始）
# 多步预测在训练数据之后使用自回归，误差会累积放大
plt.plot(time[n_train + tau:].numpy(), multistep_preds[n_train + tau:].detach().numpy(),
         label='multistep preds')
plt.xlabel('time')  # x轴标签
plt.ylabel('x')  # y轴标签
plt.xlim(1, 1000)  # x轴范围
plt.legend()  # 显示图例
plt.show()  # 显示图像

max_steps = 64  # 最大预测步数，测试从1步到64步的预测能力

# 创建特征矩阵用于k步预测分析
# 形状: (T-tau-max_steps+1, tau+max_steps) = (937, 68)
# 行数937: 从每个起始位置可以完整预测64步而不超出序列边界 (1000-4-64+1=937)
# 列数68: 前4列(tau)放真实观测，后64列放1-64步的预测值
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

# 填充前tau=4列：真实历史观测数据
# 对于第i列(0<=i<4)，存储从x[i]开始的连续937个值
# 列0: x[0:937], 列1: x[1:938], 列2: x[2:939], 列3: x[3:940]
# 每行features[r, 0:4]代表位置r+4的真实历史窗口 [x[r], x[r+1], x[r+2], x[r+3]]
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 填充后max_steps=64列：各步长的自回归预测
# 列i (4<=i<68) 存储第(i-4+1)=(i-3)步的预测值
# 例如：列4是第1步预测，列5是第2步预测，列67是第64步预测
# 每次用前tau=4列（可能是真实值或已预测值）来预测下一步
# 这是链式自回归：第k步预测基于第k-1步的预测结果
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

# 绘制不同步长的预测效果
steps = (1, 4, 16, 64)  # 定义要绘制的预测步长：1步、4步、16步、64步
plt.figure(figsize=(6, 3))
# 循环绘制每种步长的预测结果
# tau+i-1 是 features 中对应 i 步预测的列索引
# 例如：i=1时，列索引=tau+0=4，是第1步预测；i=4时，列索引=tau+3=7，是第4步预测
for i in steps:
    # 计算时间轴：从tau+i-1开始到T-max_steps+i结束
    # 每个步长的时间轴长度相同，都是 T-tau-max_steps+1 = 937
    time_slice = time[tau + i - 1: T - max_steps + i]
    # 获取对应步长的预测值列
    pred_slice = features[:, tau + i - 1].detach().numpy()
    plt.plot(time_slice.numpy(), pred_slice, label=f'{i}-step preds')
plt.xlabel('time')  # x轴标签
plt.ylabel('x')  # y轴标签
plt.xlim(5, 1000)  # x轴范围
plt.legend()  # 显示图例，标注各步长预测
plt.show()  # 显示图像