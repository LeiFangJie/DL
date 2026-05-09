import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import math


class PatchEmbedding(nn.Module):
    """使用卷积将图像转换为patch嵌入序列"""
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (Batch, Channels, Height, Width) = (B, 1, 28, 28)
        x = self.proj(x)       # (B, embed_dim=64, H/P=4, W/P=4)
        x = x.flatten(2)       # (B, 64, num_patches=16)  展平H,W维度
        x = x.transpose(1, 2)  # (B, num_patches=16, embed_dim=64)  交换维度顺序
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        # x: (Batch, Num_patches, Embed_dim) = (B, N, C)
        B, N, C = x.shape
        
        # qkv投影并拆分为多头: (B, N, 3, num_heads, head_dim) -> 3 x (B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q,k,v: (B, num_heads=4, N=16, head_dim=16)
        
        # 注意力计算: Q·K^T / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, 4, 16, 16)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和并恢复维度
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, 16, 64)
        x = self.proj(x)    # 输出投影
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """前馈神经网络"""
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for FashionMNIST"""
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10,
                 embed_dim=64, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 可学习的[class] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x: (Batch, Channels, Height, Width) = (B, 1, 28, 28)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches=16, embed_dim=64)
        
        # 添加[class] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]  # 取[class] token的输出
        logits = self.head(cls_output)
        
        return logits


def get_fashion_mnist_loaders(batch_size=128):
    """加载FashionMNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST的均值和标准差
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}')
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    # 超参数设置
    batch_size = 128
    epochs = 50
    lr = 1e-3
    
    # 模型参数 (针对28x28的FashionMNIST优化)
    img_size = 28
    patch_size = 7  # 28/7=4, 所以会有4x4=16个patches
    embed_dim = 64
    depth = 6
    num_heads = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print('加载FashionMNIST数据集...')
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size)
    
    # 创建模型
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,
        num_classes=10,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)
    
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练循环
    best_acc = 0
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        print(f'训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'vit_fashion_mnist_best.pth')
            print(f'保存最佳模型，准确率: {best_acc:.2f}%')
    
    print(f'\n训练完成! 最佳测试准确率: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
