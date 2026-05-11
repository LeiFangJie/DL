import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import math
from tqdm import tqdm
import time


# ========== 1. Tiny ImageNet 数据集 ==========

class TinyImageNetDataset(Dataset):
    """Tiny ImageNet 200类数据集"""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        # 加载类别映射
        wnid_to_idx = {}
        with open(os.path.join(root, 'wnids.txt'), 'r') as f:
            for idx, line in enumerate(f):
                wnid_to_idx[line.strip()] = idx
        self.wnid_to_idx = wnid_to_idx
        self.num_classes = len(wnid_to_idx)

        # 加载数据
        if train:
            self.data, self.targets = self._load_train_data(root, wnid_to_idx)
        else:
            self.data, self.targets = self._load_val_data(root, wnid_to_idx)

    def _load_train_data(self, root, wnid_to_idx):
        data = []
        targets = []
        train_dir = os.path.join(root, 'train')

        for wnid in sorted(os.listdir(train_dir)):
            if wnid not in wnid_to_idx:
                continue
            label = wnid_to_idx[wnid]
            class_dir = os.path.join(train_dir, wnid, 'images')
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    data.append(img_path)
                    targets.append(label)

        return data, targets

    def _load_val_data(self, root, wnid_to_idx):
        data = []
        targets = []
        val_dir = os.path.join(root, 'val')
        val_annotations = os.path.join(val_dir, 'val_annotations.txt')

        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                wnid = parts[1]
                if wnid in wnid_to_idx:
                    img_path = os.path.join(val_dir, 'images', img_name)
                    data.append(img_path)
                    targets.append(wnid_to_idx[wnid])

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]

        # 加载图像 (64x64)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target


def get_tiny_imagenet_loaders(data_root='./data/tiny-imagenet-200', batch_size=64, num_workers=4):
    """加载Tiny ImageNet数据集，将64x64 reshape到224x224"""

    # 训练时增强: 随机裁剪+水平翻转+颜色抖动
    train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证时不增强，只Resize和Normalize
    val_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TinyImageNetDataset(data_root, train=True, transform=train_transform)
    val_dataset = TinyImageNetDataset(data_root, train=False, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ========== 2. 模型构建函数 ==========

def create_finetune_model(num_classes=200):
    """创建微调模型：加载预训练Swin Transformer，替换分类头，冻结特征提取器"""
    
    # 加载ImageNet-1K v2预训练的Swin Transformer Tiny
    model = models.swin_t(weights='IMAGENET1K_V1')
    
    # 替换分类头：从1000类改为200类
    original_features = model.head.in_features
    model.head = nn.Linear(original_features, num_classes)
    
    # 冻结所有特征提取器参数
    for name, param in model.named_parameters():
        if 'head' not in name:  # 只保留分类头参数可训练
            param.requires_grad = False
    
    return model


# ========== 3. 训练函数 ==========

def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    acc = 100. * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc


def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='[Val]', ncols=100, leave=False)

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


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段: 线性增长
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine Annealing阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ========== 4. 主函数 ==========

def main():
    # 超参数设置（与参考代码保持一致）
    batch_size = 64
    num_epochs = 100
    base_lr = 5e-4
    warmup_epochs = 5
    weight_decay = 0.05
    label_smoothing = 0.1

    # 数据路径
    data_root = './data/tiny-imagenet-200'

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建微调模型
    model = create_finetune_model(num_classes=200).to(device)

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数数量: {total_params:,} ({total_params/1e6:.2f}M)')
    print(f'可训练参数数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)')

    # 损失函数 (带标签平滑)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 优化器 (只优化分类头参数)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs, base_lr, min_lr=1e-6)

    # 加载数据
    print('加载Tiny ImageNet数据集...')
    if not os.path.exists(data_root):
        print(f'错误: 数据目录不存在: {data_root}')
        print('请下载Tiny ImageNet数据集并解压到该目录')
        print('下载地址: http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        return

    train_loader, val_loader = get_tiny_imagenet_loaders(data_root, batch_size)
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'验证集大小: {len(val_loader.dataset)}')

    # 训练循环
    best_acc = 0
    print("\n" + "="*60)
    print("开始微调 Swin Transformer Tiny on Tiny ImageNet")
    print("="*60)

    for epoch in range(num_epochs):
        start_time = time.time()

        # 更新学习率
        current_lr = scheduler.step(epoch)

        # 训练和验证
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(f'\nEpoch: {epoch+1}/{num_epochs} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'swin_tiny_imagenet_finetune_best.pth')
            print(f'*** 保存最佳模型，验证准确率: {best_acc:.2f}% ***')

    print('\n' + "="*60)
    print(f'微调完成! 最佳验证准确率: {best_acc:.2f}%')
    print("="*60)


if __name__ == '__main__':
    main()