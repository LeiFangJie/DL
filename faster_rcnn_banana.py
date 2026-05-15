#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster R-CNN 香蕉检测（修复 NaN 版）
=====================================
修复内容：
  1. 图像归一化到 [0, 1]（关键！Faster R-CNN 要求输入在此范围）
  2. 分层学习率：Backbone 1e-4，检测头 5e-3
  3. 增加梯度裁剪防止爆炸
  4. 增加数据有效性检查（框坐标、面积）
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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ==================== 1. 数据集下载（与你原有代码一致） ====================

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


# ==================== 2. 修复后的数据集类 ====================

class BananaFRCNNDataset(Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'read {len(self.features)} {"training" if is_train else "validation"} examples')

    def __getitem__(self, idx):
        # 🔧 修复1：图像必须归一化到 [0, 1]！
        # torchvision.io.read_image 返回 uint8 (0-255)，除以 255 得到 [0, 1] float
        image = self.features[idx].float() / 255.0

        label = self.labels[idx]  # (1, 5) -> [class, xmin, ymin, xmax, ymax] 已归一化

        # 转回像素坐标（图像尺寸 256x256）
        boxes = label[0, 1:] * 256.0  # (4,)
        boxes = boxes.unsqueeze(0)     # (1, 4)

        # 🔧 修复2：确保框坐标有效（防止 x2<=x1 或 y2<=y1 导致面积=0）
        boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1.0)  # xmax > xmin
        boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1.0)  # ymax > ymin

        # 类别：原始是 0（香蕉），Faster R-CNN 中 0=背景，所以香蕉=1
        labels = (label[0, 0:1] + 1).long()  # (1,)

        # 计算面积
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
        DataLoader(BananaFRCNNDataset(True), batch_size=batch_size,
                   shuffle=True, collate_fn=collate_fn),
        DataLoader(BananaFRCNNDataset(False), batch_size=batch_size,
                   shuffle=False, collate_fn=collate_fn)
    )


# ==================== 3. 模型构建 ====================

def get_model(num_classes=2, pretrained=True):
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
    except Exception:
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ==================== 4. 修复后的训练循环 ====================

def train_one_epoch(model, optimizer, data_loader, device, epoch, max_norm=1.0):
    """
    训练一个 epoch，增加梯度裁剪防止 NaN。
    max_norm: 梯度裁剪阈值，通常 0.1 ~ 1.0
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 如果 loss 已经是 nan，跳过这个 batch 并报警告
        if torch.isnan(losses):
            print(f"  ⚠️  Warning: NaN detected at batch {i+1}, skipping...")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()

        # 🔧 修复3：梯度裁剪，防止 RPN 回归 loss 爆炸导致权重变 NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if (i + 1) % 10 == 0 or i == len(data_loader) - 1:
            print(f"  Epoch[{epoch}] Batch[{i+1}/{len(data_loader)}] "
                  f"Total: {losses.item():.4f} | "
                  f"RPN_cls: {loss_dict.get('loss_objectness', 0):.3f} "
                  f"RPN_reg: {loss_dict.get('loss_rpn_box_reg', 0):.3f} | "
                  f"Cls: {loss_dict.get('loss_classifier', 0):.3f} "
                  f"Reg: {loss_dict.get('loss_box_reg', 0):.3f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.train()  # 训练模式才返回 loss
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / len(data_loader)


# ==================== 5. 预测与可视化 ====================

@torch.no_grad()
def predict(image, model, device, threshold=0.5):
    """
    对单张或多张图片做预测。
    注意：Faster R-CNN eval 模式要求输入是 list of 3D tensors [C, H, W]！
    """
    model.eval()
    
    # 关键修复：不要 unsqueeze(0)！直接包成 list 即可
    if image.dim() == 3:
        # 单张图: (C, H, W) -> [(C, H, W)]
        x = [image.to(device)]
    elif image.dim() == 4:
        # Batch: (B, C, H, W) -> 拆成 B 个 (C, H, W)
        x = [t for t in image.to(device)]
    else:
        raise ValueError(f"image 必须是 3D 或 4D，当前是 {image.dim()}D")
    
    # 传入 list，模型内部自己处理
    predictions = model(x)
    pred = predictions[0]  # 取第一张图的结果
    
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
    img = (img_tensor * 255).permute(1, 2, 0).cpu().long()  # 转回 0-255 显示
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
    # ---------------- 超参数（保守设置，防止 NaN） ----------------
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    NUM_CLASSES = 2         # 香蕉 + 背景
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[INFO] 设备: {DEVICE}")
    print(f"[INFO] 类别数: {NUM_CLASSES}")

    # ---------------- 数据 ----------------
    train_loader, val_loader = load_data_bananas(BATCH_SIZE)

    # ---------------- 模型 ----------------
    print("[INFO] 加载预训练 Faster R-CNN...")
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # 🔧 修复4：分层学习率
    # Backbone 用很小的 lr（特征提取层已经很好了，只需微调）
    # 检测头（RPN + RoI Head）用稍大的 lr（新初始化的层需要多学习）
    backbone_params = []
    rpn_params = []
    roi_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'rpn' in name:
            rpn_params.append(param)
        elif 'roi_heads' in name:
            roi_params.append(param)
        else:
            other_params.append(param)

    # Backbone: 1e-4, RPN/RoI: 5e-3
    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': rpn_params, 'lr': 5e-3},
        {'params': roi_params, 'lr': 5e-3},
        {'params': other_params, 'lr': 5e-3},
    ], momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("[INFO] 使用分层学习率: Backbone 1e-4, 检测头 5e-3")
    print("[INFO] 开始训练...")

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, max_norm=1.0)
        val_loss = evaluate(model, val_loader, DEVICE)

        print(f"[Summary] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'faster_rcnn_banana_best.pth')
            print("[INFO] 验证 loss 下降，已保存最佳模型")

        lr_scheduler.step()

    torch.save(model.state_dict(), 'faster_rcnn_banana_final.pth')
    print("\n[INFO] 训练完成！")

    # ---------------- 预测 ----------------
    print("\n[INFO] 展示预测效果...")
    val_dataset = BananaFRCNNDataset(False)
    img, _ = val_dataset[0]
    output = predict(img, model, DEVICE, threshold=0.5)
    print(f"[INFO] 检测到 {len(output)} 个香蕉")
    display(img, output, threshold=0.5)


if __name__ == '__main__':
    main()
