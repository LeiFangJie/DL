import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import math
from tqdm import tqdm
import time


# ========== 1. Swin Transformer 核心组件 ==========

def window_partition(x, window_size):
    """
    将特征图分割成窗口
    x: (B, H, W, C)
    return: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口合并回特征图
    windows: (B * num_windows, window_size, window_size, C)
    return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力 (W-MSA 或 SW-MSA)"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))

        # 计算相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        x: (B * num_windows, N, C), N = window_size * window_size
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#(3,B*nun_windows,num_heads,N,C//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


    """Swin Transformer Block, 包含W-MSA和SW-MSA"""
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, H, W, attn_mask=None):
        """
        x: (B, H*W, C)
        attn_mask: 移位窗口注意力的掩码
        """
        B, L, C = x.shape
        assert L == H * W, "输入特征维度不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 窗口分割
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 窗口合并
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer，用于下采样"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "输入特征维度不匹配"
        assert H % 2 == 0 and W % 2 == 0, f"H和W必须是偶数，当前H={H}, W={W}"

        x = x.view(B, H, W, C)

        # 分割成4个子区域
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def create_mask(H, W, window_size, shift_size, device):
    """为移位窗口注意力创建掩码"""
    if shift_size == 0:
        return None

    # 创建循环移位后的区域标记图
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)#(B, num_patches=56*56, embed_dim)
        x = self.norm(x)
        return x

#=================================================================================================================
class BasicLayer(nn.Module):
    """Swin Transformer的一个Stage"""
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        # 构建Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])

        # 下采样层
        self.downsample = downsample(dim=dim) if downsample is not None else None

    def forward(self, x, H, W):
        # 为移位窗口生成掩码 (只在需要时生成一次)
        attn_mask = None
        device = x.device

        for i, blk in enumerate(self.blocks):
            if blk.shift_size > 0 and attn_mask is None:
                attn_mask = create_mask(H, W, blk.window_size, blk.shift_size, device)
            x = blk(x, H, W, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down
        else:
            return x, H, W, x


class SwinTransformer(nn.Module):
    """Swin Transformer Tiny for Tiny ImageNet"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=200,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))#768
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches#56
        patches_resolution = self.patch_embed.patches_resolution#56，56
        self.patches_resolution = patches_resolution

        # 绝对位置嵌入 (不使用，因为Swin用相对位置偏置)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建Stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):# i_layer个stage
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),#96，192，384，768
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        H, W = self.patches_resolution
        for layer in self.layers:
            x, H, W, x_out = layer(x, H, W)
            x = x_out

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ========== 2. Tiny ImageNet 数据集 ==========

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
    # 超参数设置
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

    # 创建模型: Swin Transformer Tiny
    # 参数量约 28M
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=200,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数数量: {num_params:,} ({num_params/1e6:.2f}M)')

    # 损失函数 (带标签平滑)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 优化器 (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

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
    print("开始训练 Swin Transformer Tiny on Tiny ImageNet")
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
            torch.save(model.state_dict(), 'swin_tiny_imagenet_best.pth')
            print(f'*** 保存最佳模型，验证准确率: {best_acc:.2f}% ***')

    print('\n' + "="*60)
    print(f'训练完成! 最佳验证准确率: {best_acc:.2f}%')
    print("="*60)


if __name__ == '__main__':
    main()
