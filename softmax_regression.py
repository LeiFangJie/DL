import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

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

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier初始化
        nn.init.zeros_(m.bias)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 3
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


for epoch in range(num_epochs):
    total_loss = 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += l.item()
    train_acc = evaluate_accuracy(net, train_iter)
    test_acc = evaluate_accuracy(net, test_iter)  # 添加测试集评估
    avg_loss = total_loss / len(train_iter)
    print(f"epoch {epoch + 1}, loss: {avg_loss:.4f}, train_acc: {train_acc:.2f}%, test_acc: {test_acc:.2f}%")


