import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== 1. 数据预处理 ==========
# CIFAR-10 图像是 32x32，ResNet18 预训练模型是在 224x224 ImageNet 上训练的
# 因此需要 resize 到 224x224

train_transform = transforms.Compose([
    transforms.Resize(224),  # ResNet18 需要 224x224 输入
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 预训练的标准化参数
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ========== 2. 加载预训练 ResNet18 并修改 ==========
# 加载预训练权重
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 冻结所有层（可选策略1：特征提取器模式）
# for param in model.parameters():
#     param.requires_grad = False

# 修改最后的全连接层：原来输出 1000 类(ImageNet)，改为 10 类(CIFAR-10)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(device)


# ========== 3. 训练设置 ==========
criterion = nn.CrossEntropyLoss()

# 只对修改后的 fc 层使用较大学习率，其他层使用较小学习率（分层学习率）
# 这是微调的关键技巧
finetune_lr = 0.001  # 微调层的学习率
feature_lr = 1e-5    # 特征提取层的学习率（可选，如果冻结则为0）

# 分组参数
fc_params = []
feature_params = []

for name, param in model.named_parameters():
    if 'fc' in name:  # 最后一层全连接层
        fc_params.append(param)
    else:
        feature_params.append(param)

optimizer = optim.Adam([
    {'params': feature_params, 'lr': feature_lr},
    {'params': fc_params, 'lr': finetune_lr}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# ========== 4. 训练和测试函数 ==========
def train_epoch(model, loader, criterion, optimizer, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm 包装 dataloader，显示当前 epoch 进度
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', ncols=100)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 实时更新进度条后缀信息
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    acc = 100. * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc


def test(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # tqdm 包装测试过程
    pbar = tqdm(loader, desc='[Test]', ncols=100, leave=False)

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})

    acc = 100. * correct / total
    avg_loss = test_loss / len(loader)
    return avg_loss, acc


# ========== 5. 主训练循环 ==========
num_epochs = 30
best_acc = 0

print("\n" + "="*50)
print("开始训练 ResNet18 on CIFAR-10")
print("="*50)

for epoch in range(num_epochs):
    start_time = time.time()

    print(f'\nEpoch: {epoch+1}/{num_epochs}')
    print('-' * 30)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs)
    test_loss, test_acc = test(model, test_loader, criterion)

    scheduler.step()

    epoch_time = time.time() - start_time

    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Test Loss:  {test_loss:.4f}  | Test Acc:  {test_acc:.2f}%')
    print(f'Time: {epoch_time:.1f}s')

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_resnet18_cifar10.pth')
        print(f'*** 保存最佳模型，准确率: {best_acc:.2f}% ***')

print('\n' + "="*50)
print(f'训练完成! 最佳测试准确率: {best_acc:.2f}%')
print("="*50)
