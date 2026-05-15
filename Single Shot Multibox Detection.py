"""
TinySSD 目标检测（PyTorch 原生实现，不依赖 d2l）
用于香蕉检测数据集
"""

import os
import hashlib
import urllib.request
import zipfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ==================== 1. 数据集下载与加载 ====================

DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB = {
    'banana-detection': (
        DATA_URL + 'banana-detection.zip',
        '5de26c8fce5ccdea9f91267273464dc968d20d72'
    )
}


def download_extract(name, cache_dir='../data'):
    """下载并解压数据集，返回解压后的文件夹路径"""
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # 若文件已存在且校验通过，直接返回
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
    """读取香蕉检测数据集图像与标签"""
    data_dir = download_extract('banana-detection')
    split = 'bananas_train' if is_train else 'bananas_val'
    csv_path = os.path.join(data_dir, split, 'label.csv')
    csv_data = pd.read_csv(csv_path).set_index('img_name')

    images, targets = [], []
    img_dir = os.path.join(data_dir, split, 'images')
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(img_dir, img_name)))
        targets.append(list(target))
    # 坐标归一化到 [0, 1]（原图尺寸为 256×256）
    return images, torch.tensor(targets).unsqueeze(1) / 256


class BananasDataset(Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'read {len(self.features)} {"training" if is_train else "validation"} examples')

    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    """加载训练集与验证集"""
    return (
        DataLoader(BananasDataset(True), batch_size, shuffle=True),
        DataLoader(BananasDataset(False), batch_size)
    )


# ==================== 2. 坐标转换工具 ====================

def box_corner_to_center(boxes):
    """角点格式 (xmin, ymin, xmax, ymax) -> 中心格式 (cx, cy, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_center_to_corner(boxes):
    """中心格式 (cx, cy, w, h) -> 角点格式 (xmin, ymin, xmax, ymax)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


# ==================== 3. 锚框生成 ====================

def multibox_prior(data, sizes, ratios):
    """
    以特征图每个像素为中心生成锚框。
    返回: (1, H*W*boxes_per_pixel, 4)，坐标为角点格式 (xmin, ymin, xmax, ymax)，归一化到 [0,1]
    """
    in_h, in_w = data.shape[-2:]
    device = data.device
    num_sizes, num_ratios = len(sizes), len(ratios)
    # SSD 组合规则：所有 sizes 配 ratio[0]；size[0] 配其余 ratios
    boxes_per_pixel = num_sizes + num_ratios - 1

    size_t = torch.tensor(sizes, device=device)
    ratio_t = torch.tensor(ratios, device=device)

    # 每个像素的中心坐标，归一化到 [0, 1]
    step_h, step_w = 1.0 / in_h, 1.0 / in_w
    center_h = (torch.arange(in_h, device=device) + 0.5) * step_h
    center_w = (torch.arange(in_w, device=device) + 0.5) * step_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 组合生成所有锚框的半宽、半高
    w = torch.cat([
        size_t * torch.sqrt(ratio_t[0]),
        sizes[0] * torch.sqrt(ratio_t[1:])
    ]) * in_h / in_w  # 修正非正方形图像的宽高比
    h = torch.cat([
        size_t / torch.sqrt(ratio_t[0]),
        sizes[0] / torch.sqrt(ratio_t[1:])
    ])

    # 每个中心点复制 boxes_per_pixel 份，与半宽高偏移相加得到角点坐标
    half = torch.stack([-w, -h, w, h], dim=1) / 2              # (boxes_per_pixel, 4)
    half = half.repeat(in_h * in_w, 1)                         # (H*W*boxes_per_pixel, 4)
    centers = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
    centers = centers.repeat_interleave(boxes_per_pixel, dim=0)

    return (centers + half).unsqueeze(0)


# ==================== 4. 网络定义 ====================

def cls_predictor(in_channels, num_anchors, num_classes):
    """类别预测头：输出通道 = 锚框数 × (类别数+1)，其中 +1 为背景"""
    return nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(in_channels, num_anchors):
    """边界框预测头：输出通道 = 锚框数 × 4（中心偏移 + 宽高缩放）"""
    return nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)


def down_sample_blk(in_c, out_c):
    """下采样块：2×(卷积+BN+ReLU) + MaxPool"""
    blk = []
    for _ in range(2):
        blk.extend([
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ])
        in_c = out_c
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    """基础网络：3->16->32->64，每步都下采样"""
    filters = [3, 16, 32, 64]
    return nn.Sequential(*[down_sample_blk(filters[i], filters[i + 1]) for i in range(len(filters) - 1)])


def get_blk(i):
    """获取第 i 个模块（5 层检测分别对应不同下采样策略）"""
    if i == 0:
        return base_net()               # 3->16->32->64
    if i == 1:
        return down_sample_blk(64, 128)
    if i == 4:
        return nn.AdaptiveMaxPool2d((1, 1))  # 全局池化
    return down_sample_blk(128, 128)    # i=2,3


def flatten_pred(pred):
    """将 (B, C, H, W) 的预测结果展平为 (B, H*W*C_per_position)"""
    # 先把通道维度移到最后 -> (B, H, W, C)，再展平后两维
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    """拼接多层预测结果，沿锚框数量维度合并"""
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def blk_forward(X, blk, size, ratio, cls_pred, bbox_pred):
    """单步前向：特征提取 -> 生成锚框 -> 类别预测 + 框偏移预测"""
    Y = blk(X)
    anchors = multibox_prior(Y, size, ratio)
    return Y, anchors, cls_pred(Y), bbox_pred(Y)


# 5 层检测的超参数：每层的锚框尺寸与长宽比
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1  # 2 + 3 - 1 = 4


class TinySSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        in_channels = [64, 128, 128, 128, 128]
        # 动态创建 5 组 (blk, cls, bbox) 模块
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
        # 拼接 5 层结果：锚框沿 dim=1 合并；预测结果展平后 reshape 出语义维度
        anchors = torch.cat(anchors, dim=1)                     # (B, 总锚框数, 4)
        
        # 类别预测 reshape: (B, 总锚框数, num_classes+1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        
        # 边界框预测也要 reshape: (B, 总锚框数, 4)，否则后续无法与掩码对齐
        bbox_preds = concat_preds(bbox_preds)
        bbox_preds = bbox_preds.reshape(bbox_preds.shape[0], -1, 4)
        
        return anchors, cls_preds, bbox_preds


# ==================== 5. 训练辅助：标签分配与损失 ====================

def multibox_target(anchors, labels):
    """
    为所有锚框分配真实标签。
    参数:
        anchors: (1, num_anchors, 4) 角点坐标
        labels:  (B, num_gt, 5)      [class, xmin, ymin, xmax, ymax]
    返回:
        bbox_labels: (B, num_anchors, 4)  偏移量真值
        bbox_masks:  (B, num_anchors, 4)  正样本掩码（1表示参与定位损失）
        cls_labels:  (B, num_anchors)     类别标签（0为背景）
    """
    anchors = anchors.squeeze(0)                     # (num_anchors, 4)
    num_anchors = anchors.shape[0]
    device = anchors.device
    batch_size = labels.shape[0]

    # 初始化输出
    bbox_labels = torch.zeros((batch_size, num_anchors, 4), device=device)
    bbox_masks = torch.zeros((batch_size, num_anchors, 4), device=device)
    cls_labels = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=device)

    # 转中心格式 (cx, cy, w, h)，用于 SSD 偏移编码
    anchors_cxcywh = box_corner_to_center(anchors)

    for i in range(batch_size):
        label = labels[i]
        valid = label[:, 0] >= 0                     # 过滤无效占位标签
        if not valid.any():
            continue

        gt_boxes = label[valid, 1:]                 # (num_gt, 4) 角点
        gt_classes = label[valid, 0].long() + 1      # 类别从 1 开始（0 预留给背景）
        num_gt = gt_boxes.shape[0]

        # 计算 IoU（使用 torchvision 高级 API，输入角点格式）
        iou = torchvision.ops.box_iou(gt_boxes, anchors)  # (num_gt, num_anchors)

        # 每个锚框最匹配的 GT
        gt_idx_per_anchor = iou.argmax(dim=0)         # (num_anchors,)
        max_iou_per_anchor = iou[gt_idx_per_anchor, torch.arange(num_anchors)]

        # 每个 GT 最匹配的锚框（强制为正样本，防止漏标）
        anchor_idx_per_gt = iou.argmax(dim=1)         # (num_gt,)

        # 正样本：IoU >= 0.5，或 是某 GT 的最佳匹配锚框
        pos_mask = max_iou_per_anchor >= 0.5
        pos_mask[anchor_idx_per_gt] = True

        # 分配类别标签
        matched_gt_idx = gt_idx_per_anchor[pos_mask]
        cls_labels[i, pos_mask] = gt_classes[matched_gt_idx]

        # 计算偏移真值（SSD 标准编码公式）
        gt_matched = gt_boxes[matched_gt_idx]
        gt_cxcywh = box_corner_to_center(gt_matched)
        anc = anchors_cxcywh[pos_mask]

        offset_xy = (gt_cxcywh[:, :2] - anc[:, :2]) / anc[:, 2:]
        offset_wh = torch.log(gt_cxcywh[:, 2:] / anc[:, 2:] + 1e-6)

        bbox_labels[i, pos_mask, :2] = offset_xy
        bbox_labels[i, pos_mask, 2:] = offset_wh
        bbox_masks[i, pos_mask] = 1.0                # 只有正样本参与定位损失

    return bbox_labels, bbox_masks, cls_labels


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, alpha=3.0):
    """总损失 = 分类损失 + alpha * 定位损失"""
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]

    cls = F.cross_entropy(cls_preds.reshape(-1, num_classes),
                          cls_labels.reshape(-1),
                          reduction='none').reshape(batch_size, -1).mean(dim=1)

    bbox = F.l1_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks,
                     reduction='none').mean(dim=(1, 2))
    
    return cls + alpha * bbox  # 🔥 定位损失权重放大 2 倍

def cls_eval(cls_preds, cls_labels):
    """计算分类正确的样本数"""
    return (cls_preds.argmax(dim=-1) == cls_labels).sum().item()


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """计算正样本上的边界框绝对误差之和"""
    return torch.abs((bbox_labels - bbox_preds) * bbox_masks).sum().item()


# ==================== 6. 推理：NMS 与可视化 ====================

def multibox_detection(cls_probs, offset_preds, anchors,
                       nms_threshold=0.2, pos_threshold=0.01):
    """
    后处理：将网络输出解码为最终检测框，并做 NMS 去重。
    参数:
        cls_probs:    (B, num_anchors, num_classes)  softmax 后的类别概率
        offset_preds: (B, num_anchors, 4)            预测的偏移量
        anchors:      (1, num_anchors, 4)            角点坐标
    返回:
        output: (B, num_keep, 6) -> [class, score, xmin, ymin, xmax, ymax]
    """
    anchors = anchors.squeeze(0)
    anchors_cxcywh = box_corner_to_center(anchors)
    batch_size = cls_probs.shape[0]
    device = cls_probs.device

    output = []
    for i in range(batch_size):
        # 解码偏移量：中心坐标 + 指数宽高 -> 再转回角点
        pred_cxcy = anchors_cxcywh[:, :2] + offset_preds[i, :, :2] * anchors_cxcywh[:, 2:]
        pred_wh = anchors_cxcywh[:, 2:] * torch.exp(offset_preds[i, :, 2:])
        pred_boxes = box_center_to_corner(torch.cat([pred_cxcy, pred_wh], dim=-1))

        # 每个锚框的最佳类别与置信度
        prob, label = cls_probs[i].max(dim=-1)

        # 过滤背景（label=0）与低置信度
        mask = (label > 0) & (prob > pos_threshold)
        if not mask.any():
            output.append(torch.zeros((0, 6), device=device))
            continue

        keep_boxes = pred_boxes[mask]
        keep_scores = prob[mask]
        keep_labels = label[mask].float()

        # 使用 torchvision batched_nms：按类别分组做非极大值抑制（高级 API）
        keep_indices = torchvision.ops.batched_nms(
            keep_boxes, keep_scores, keep_labels.long(), nms_threshold
        )

        res = torch.cat([
            keep_labels[keep_indices].unsqueeze(1),
            keep_scores[keep_indices].unsqueeze(1),
            keep_boxes[keep_indices]
        ], dim=1)
        output.append(res)

    # 填充为统一长度（兼容原接口，无效框 class=-1）
    max_len = max(len(o) for o in output) if output else 0
    padded = []
    for o in output:
        pad = torch.zeros((max_len - len(o), 6), device=device)
        pad[:, 0] = -1
        padded.append(torch.cat([o, pad], dim=0))
    return torch.stack(padded)


def predict(X, net, device):
    """对单张图片做预测，返回过滤后的检测框"""
    net.eval()
    with torch.no_grad():
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=-1)
        output = multibox_detection(cls_probs, bbox_preds, anchors)
    # 去掉填充的无效框（class == -1）
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx] if idx else torch.zeros((0, 6), device=device)


def display(img, output, threshold=0.9):
    """在图像上绘制检测结果"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    h, w = img.shape[:2]

    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        # 归一化坐标 -> 像素坐标
        bbox = row[2:6] * torch.tensor([w, h, w, h], device=row.device)
        x1, y1, x2, y2 = bbox.cpu().numpy()

        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{score:.2f}', color='white', fontsize=12, va='top')

    ax.axis('off')
    plt.show()


# ==================== 7. 训练循环 ====================

if __name__ == '__main__':
    batch_size = 32
    train_iter, _ = load_data_bananas(batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = TinySSD(num_classes=1).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    num_epochs = 50
    net.train()

    for epoch in range(num_epochs):
        # metric: [分类正确数, 分类总数, 框误差和, 正样本数]
        metric = [0.0] * 4
        start = time.time()

        for features, target in train_iter:
            optimizer.zero_grad()
            X, Y = features.to(device), target.to(device)

            # 前向：生成锚框 + 类别预测 + 框偏移预测
            anchors, cls_preds, bbox_preds = net(X)

            # 标签分配：根据 GT 为每个锚框打标签（正/负样本）
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)

            # 计算损失并反向传播
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            optimizer.step()

            # 统计指标
            metric[0] += cls_eval(cls_preds, cls_labels)
            metric[1] += cls_labels.numel()
            metric[2] += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            metric[3] += bbox_masks.sum().item()

        cls_err = 1 - metric[0] / metric[1]
        bbox_mae = metric[2] / metric[3]
        print(f'Epoch {epoch + 1}: class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}, '
              f'time {time.time() - start:.1f}s')

    # ==================== 8. 预测示例 ====================
    X = torchvision.io.read_image(r"D:\FAFU_work\data\banana-detection\bananas_val\images\30.png").unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
    output = predict(X, net, device)
    display(img, output.cpu(), threshold=0.8)