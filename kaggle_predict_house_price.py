import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

#@save
DATA_HUB = dict() #key为数据集名称，value为二元组(url,sha-1密钥)
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('D:/FAFU_work/DL/data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname#返回文件下载路径

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)#文件路径
    base_dir = os.path.dirname(fname)#文件所在目录
    data_dir, ext = os.path.splitext(fname)#文件名和扩展名
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')#打开zip文件
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')#打开tar文件
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)#解压文件
    return os.path.join(base_dir, folder) if folder else data_dir#返回解压后的文件夹路径

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)#下载所有文件

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)



#==================================数据下载==================================

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

#print(f'data_hub:{DATA_HUB}')

#下载数据并进行数据清洗
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(f'训练数据:')
print(train_data.head())
print(f'训练数据形状: {train_data.shape}')
print(f'测试数据:')
print(test_data.head())
print(f'测试数据形状: {test_data.shape}')
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))#屏蔽d和最后一行价格并将他们拼接

#================================数据预处理====================================

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = []
for col in all_features.columns:
    try:
        # 尝试计算均值，成功则是数值列
        all_features[col].mean()
        numeric_features.append(col)
    except:
        # 失败则是非数值列，跳过
        continue

numeric_features = pd.Index(numeric_features)
print(f"筛选出数值特征: {len(numeric_features)}列")

# 标准化数值特征
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# "Dummy_na=True"将"na"（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)

# 确保所有数据都是数值类型
all_features = all_features.astype(np.float32)
print(f"one-hot编码后的特征: {all_features.shape}")
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
print(f"数据类型检查: {all_features.dtypes.unique()}")
print(f"训练数据形状: {train_features.shape}")
print(f"测试数据形状: {test_features.shape}")
print(f"训练标签形状: {train_labels.shape}")
print(f"训练数据:\n{train_features}")
print(f"测试数据:\n{test_features}")
print(f"训练标签:\n{train_labels}")
print("数据预处理完成！")
# ========================模型-损失函数-优化算法====================================================

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )
    
    # Kaiming初始化 - 适用于ReLU激活函数
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    return net

def log_rmse(net, features, labels):#损失函数log_rmse
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1，大于1e6的值设置为1e6
    preds = net(features)
    # 确保预测值和标签都是正数
    clipped_preds = torch.clamp(preds, 1, float('inf'))
    clipped_labels = torch.clamp(labels, 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(clipped_labels)))
    return rmse.item()

def r2_score(net, features, labels):
    """计算R²决定系数"""
    with torch.no_grad():
        preds = net(features)
        ss_res = torch.sum((labels - preds) ** 2)
        ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_r2, test_r2 = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        train_r2.append(r2_score(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
            test_r2.append(r2_score(net, test_features, test_labels))
    return train_ls, test_ls, train_r2, test_r2

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_loss_sum, valid_loss_sum = 0, 0
    train_r2_sum, valid_r2_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls, train_r2, valid_r2 = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_loss_sum += train_ls[-1]
        valid_loss_sum += valid_ls[-1]
        train_r2_sum += train_r2[-1]
        valid_r2_sum += valid_r2[-1]
        print(f'折{i+1}：训练log_rmse={float(train_ls[-1]):.6f}, R²={float(train_r2[-1]):.4f}, '
              f'验证log_rmse={float(valid_ls[-1]):.6f}, R²={float(valid_r2[-1]):.4f}')
    return (train_loss_sum / k, valid_loss_sum / k, 
            train_r2_sum / k, valid_r2_sum / k)

# ========================训练和预测====================================
# 设置超参数
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.001, 1e-4, 32

# 进行K折交叉验证
train_loss_mean, valid_loss_mean, train_r2_mean, valid_r2_mean = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'\n{k}折交叉验证结果：')
print(f'平均训练log_rmse: {float(train_loss_mean):.6f}, R²: {float(train_r2_mean):.4f}')
print(f'平均验证log_rmse: {float(valid_loss_mean):.6f}, R²: {float(valid_r2_mean):.4f}')

# 使用全部训练数据训练最终模型并在测试集上预测
print('\n使用全部训练数据训练最终模型...')
final_net = get_net()
train_ls, _, train_r2, _ = train(final_net, train_features, train_labels, None, None,
                                 num_epochs, lr, weight_decay, batch_size)
print(f'最终模型 - 训练log_rmse: {float(train_ls[-1]):.6f}, R²: {float(train_r2[-1]):.4f}')

# 在测试集上预测
preds = final_net(test_features).detach().numpy()
test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)

print(f'\n预测完成！结果已保存到submission.csv')
print(f'预测结果示例：')
print(submission.head(10))

