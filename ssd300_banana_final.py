#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSD300 香蕉检测（预训练微调版 - 修复 module_list 属性名）
===========================================================
修复：SSDClassificationHead 的属性是 module_list（不是 cls_blocks）。
核心思路：只替换 module_list 里每个 Conv2d 的输出通道，保留整个 head 对象。
"""

import os
import hashlib
import urllib.request
import zipfile
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import ssd300_vgg16
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ==================== 1. 数据集下载 ====================

DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB = {
    'banana-detection': (
        DATA_URL + 'banana-detection.zip',
        '5de26c8fce5ccdea9f91267273464dc968d20d72'
    )
}


def download_extract(name, cache_dir='../data'):
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while chunk := f.read(1048576):
                sha1.update(chunk)
        if sha1.hexdigest() == sha1_hash:
            return os.path.splitext(fname)[0]

    print(f'Downloading {fname}...')
    urllib.request.urlretrieve(url, fname)
    with zipfile.ZipFile(fname, 'r') as z:
        z.extractall(os.path.dirname(fname))
    return os.path.splitext(fname)[0]


def read_data_bananas(is_train=True):
    data_dir = download_extract('banana-detection')
    split = 'bananas_train' if is_train else 'bananas_val'
    csv_path = os.path.join(data_dir, split, 'label.csv')
    csv_data = pd.read_csv(csv_path).set_index('img_name')

    images, targets = [], []
    img_dir = os.path.join(data_dir, split, 'images')
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(img_dir, img_name)))
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


# ==================== 2. 数据集类 ====================

class BananaSSDDataset(Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'read {len(self.features)} {"training" if is_train else "validation"} examples')

    def __getitem__(self, idx):
        image = self.features[idx].float() / 255.0
        label = self.labels[idx]

        boxes = label[0, 1:] * 256.0
        boxes = boxes.unsqueeze(0)
        boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1.0)
        boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1.0)

        labels = (label[0, 0:1] + 1).long()
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((1,), dtype=torch.int64)
        }
        return image, target

    def __len__(self):
        return len(self.features)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_data_bananas(batch_size):
    return (
        DataLoader(BananaSSDDataset(True), batch_size=batch_size,
                   shuffle=True, collate_fn=collate_fn),
        DataLoader(BananaSSDDataset(False), batch_size=batch_size,
                   shuffle=False, collate_fn=collate_fn)
    )


# ==================== 3. 模型构建（修复版） ====================

def replace_cls_head(model, num_classes=2):
    """
    只替换 SSDClassificationHead 里 module_list 的 Conv2d 输出通道。
    SSDClassificationHead 继承自 SSDScoringHead，属性名是 module_list。
    """
    old_head = model.head.classification_head

    # 读取旧类别数（COCO 预训练通常是 91）
    old_num_classes = getattr(old_head, 'num_columns', 91)  # num_columns 是 SSDScoringHead 的属性

    print(f"[INFO] 旧分类头类别数: {old_num_classes}")
    print(f"[INFO] 开始替换 {len(old_head.module_list)} 个 Conv2d...")

    for i, old_conv in enumerate(old_head.module_list):
        if not isinstance(old_conv, nn.Conv2d):
            # 如果 module 是 Sequential 等容器，递归找最后一个 Conv2d
            convs = [m for m in old_conv.modules() if isinstance(m, nn.Conv2d)]
            if not convs:
                raise RuntimeError(f"module_list[{i}] 中找不到 Conv2d")
            old_conv = convs[-1]

            # 计算 num_anchors
            num_anchors = old_conv.out_channels // old_num_classes

            # 创建新 Conv2d
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                num_anchors * num_classes,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None)
            )

            # 替换容器中的 Conv2d
            for name, module in old_conv.named_modules():
                if module is old_conv:
                    parts = name.split('.')
                    parent = old_conv
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], new_conv)
                    break
        else:
            # module_list[i] 本身就是 Conv2d（最常见情况）
            num_anchors = old_conv.out_channels // old_num_classes

            new_conv = nn.Conv2d(
                old_conv.in_channels,
                num_anchors * num_classes,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None)
            )

            old_head.module_list[i] = new_conv

        print(f"  Conv2d {i}: in={old_conv.in_channels}, anchors={num_anchors}, "
              f"out_old={old_conv.out_channels}, out_new={num_anchors * num_classes}")

    # 更新 num_columns（forward 里的 reshape 会用到）
    old_head.num_columns = num_classes

    print(f"[INFO] 分类头替换完成，新类别数: {num_classes}")
    return model


def get_model(num_classes=2, pretrained=True):
    """加载预训练 SSD300，只改分类头的输出通道。"""
    try:
        from torchvision.models.detection import SSD300_VGG16_Weights
        weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
        model = ssd300_vgg16(weights=weights)
    except Exception:
        model = ssd300_vgg16(pretrained=pretrained)

    model = replace_cls_head(model, num_classes=num_classes)
    return model


# ==================== 4. 训练循环 ====================

def train_one_epoch(model, optimizer, data_loader, device, epoch, max_norm=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses):
            print(f"  ⚠️  Warning: NaN at batch {i+1}, skipping...")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if (i + 1) % 10 == 0 or i == len(data_loader) - 1:
            print(f"  Epoch[{epoch}] Batch[{i+1}/{len(data_loader)}] "
                  f"Total: {losses.item():.4f} | "
                  f"Cls: {loss_dict.get('classification', 0):.3f} "
                  f"Reg: {loss_dict.get('bbox_regression', 0):.3f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss for loss in loss_dict.values()).item()

    model.train()
    return total_loss / len(data_loader)


# ==================== 5. 预测与可视化 ====================

@torch.no_grad()
def predict(image, model, device, threshold=0.5):
    model.eval()

    if image.dim() == 3:
        x = [image.to(device)]
    elif image.dim() == 4:
        x = [t for t in image.to(device)]
    else:
        raise ValueError(f"image 必须是 3D 或 4D，当前是 {image.dim()}D")

    predictions = model(x)
    pred = predictions[0]

    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return torch.zeros((0, 6))

    res = torch.cat([
        labels.unsqueeze(1).float(),
        scores.unsqueeze(1),
        boxes
    ], dim=1)
    return res


def display(img_tensor, output, threshold=0.5):
    img = (img_tensor * 255).permute(1, 2, 0).cpu().long()
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)

    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        x1, y1, x2, y2 = row[2:6].cpu().numpy()
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'banana {score:.2f}',
                color='white', fontsize=12, va='top')

    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ==================== 6. 主函数 ====================

def main():
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    NUM_CLASSES = 2         # 香蕉 + 背景
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] 设备: {DEVICE}")
    print(f"[INFO] 类别数: {NUM_CLASSES}")
    print("[INFO] 模型: SSD300 VGG16（只替换分类头输出通道）")

    train_loader, val_loader = load_data_bananas(BATCH_SIZE)

    print("[INFO] 加载预训练 SSD300...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    print("[INFO] 模型加载完成")

    # 分层学习率
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params,     'lr': 1e-3},
    ], momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("[INFO] 使用分层学习率: Backbone 1e-4, Head 1e-3")
    print("[INFO] 开始训练...")

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, max_norm=1.0)
        val_loss = evaluate(model, val_loader, DEVICE)

        print(f"[Summary] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ssd300_banana_best.pth')
            print("[INFO] 验证 loss 下降，已保存最佳模型")

        lr_scheduler.step()

    torch.save(model.state_dict(), 'ssd300_banana_final.pth')
    print("\n[INFO] 训练完成！")

    # 预测展示
    print("\n[INFO] 展示预测效果...")
    val_dataset = BananaSSDDataset(False)
    img, _ = val_dataset[0]
    output = predict(img, model, DEVICE, threshold=0.5)
    print(f"[INFO] 检测到 {len(output)} 个香蕉")
    display(img, output, threshold=0.5)


if __name__ == '__main__':
    main()
