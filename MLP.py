import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

def evaluate_accuracy(model, data_loader):
    """计算给定数据加载器的准确率"""
    model.eval()  # 设为评估模式
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for X, y in data_loader:
            y_hat = model(X)
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    model.train()  # 恢复训练模式
    return 100 * correct / total

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="D:/FAFU_work/DL/data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="D:/FAFU_work/DL/data", train=False, transform=trans, download=True)

#print(mnist_train[0])#查看数据情况

batch_size = 256    
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=0)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=0)
#模型定义
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(256, 10))
#初始化权重和偏置
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)

for epoch in range(num_epochs):
    epoch_loss = 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        epoch_loss += l.item()
    train_acc = evaluate_accuracy(net, train_iter)
    test_acc = evaluate_accuracy(net, test_iter)
    print('epoch %d, loss %f, train_acc %f, test_acc %f' % (epoch+1, epoch_loss/len(train_iter), train_acc, test_acc))