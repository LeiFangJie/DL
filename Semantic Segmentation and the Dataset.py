import os
import torch
import torchvision
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import display
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# 自动下载并解压到指定目录
# year='2012', image_set='trainval' 对应 VOCtrainval_11-May-2012.tar
dataset = VOCSegmentation(
    root='../data',           # 下载目录
    year='2012',
    image_set='trainval',
    download=False,           # 自动下载
    transform=None,
    target_transform=None
)

# 数据根目录路径
voc_dir = '../data/VOCdevkit/VOC2012'

#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels


#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label#return成一行列表[0,0,...1,0,2...]之类的，有256**3个元素，
    #对应label的rgb查找index对应的值应该是背景，还是物体

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])#整张表算label的rgb对应的索引是什么值
    return colormap2label[idx]#整张图rgb索引值查找是背景还是物体-他是个标签矩阵

# y = voc_label_indices(train_labels[0], voc_colormap2label())
# print(y[105:115, 130:140], VOC_CLASSES[1])

#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# imgs = []
# n = 5
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

# imgs = [img.permute(1, 2, 0) for img in imgs]
# display_imgs = imgs[::2] + imgs[1::2]

# fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
# for i, ax in enumerate(axes.flat):
#     img = display_imgs[i]
#     # 标签图通常是单通道 (H, W, 1)，需要 squeeze
#     if img.ndim == 3 and img.shape[-1] == 1:
#         img = img.squeeze(-1)
#     # 单通道用灰度显示
#     if img.ndim == 2:
#         ax.imshow(img, cmap='gray', vmin=0, vmax=255)
#     else:
#         # 如果是 uint8 (0-255)，matplotlib 可以直接显示；
#         # 如果是 float 且值大于 1，归一化到 0-1
#         if img.max() > 1.0:
#             img = img.float() / 255.0
#         ax.imshow(img)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# crop_size = (320, 480)
# voc_train = VOCSegDataset(True, crop_size, voc_dir)
# voc_test = VOCSegDataset(False, crop_size, voc_dir)

# batch_size = 64
# train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
#                                     drop_last=True,
#                                     num_workers=0)
# for X, Y in train_iter:
#     print(X.shape)
#     print(Y.shape)
#     break

#@save
def load_data_voc(batch_size, crop_size, voc_dir='../data/VOCdevkit/VOC2012', num_workers=0):
    """Load the VOC semantic segmentation dataset."""
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter#没有数据泄露，再read_voc_images函数里有定义train.txt和val.txt来读取不同的文件


#====================model ========================
pretrained_net = torchvision.models.resnet18(weights=("pretrained", torchvision.models.ResNet18_Weights.DEFAULT))

net = nn.Sequential(*list(pretrained_net.children())[:-2])
#print(list(net.children()))

X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)

num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))#21个kernel1*1卷积将通道融合，输出就是(B,21,H/32,W/32)
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

X = torch.rand(size=(1, 3, 320, 480))
print(net(X).shape)

#用双线性插值初始化转置卷积
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)


# ==================== 工具类（替代 d2l.Timer / Accumulator / Animator） ====================

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def sum(self):
        return sum(self.times)

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [
            a + (b.detach().item() if isinstance(b, torch.Tensor) else float(b))
            for a, b in zip(self.data, args)
        ]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if not device:
        device = next(iter(net.parameters())).device
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    return l.sum().detach(), accuracy(pred, y)

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=None):
    if devices is None:
        devices = try_all_gpus()
    
    timer = Timer()
    
    if len(devices) > 1:
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    else:
        net = net.to(devices[0])
    
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        metric = Accumulator(4)
        pbar = tqdm(train_iter, desc=f"Train [{epoch+1}/{num_epochs}]", leave=False)
        
        for features, labels in pbar:
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            
            pbar.set_postfix({
                'loss': f'{metric[0]/metric[2]:.3f}',
                'train_acc': f'{metric[1]/metric[3]:.3f}'
            })
        
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        tqdm.write(f'[Epoch {epoch+1}/{num_epochs}] '
                   f'loss {metric[0]/metric[2]:.3f}, '
                   f'train acc {metric[1]/metric[3]:.3f}, '
                   f'test acc {test_acc:.3f}')
    
    print(f'\nFinal: loss {metric[0]/metric[2]:.3f}, train acc {metric[1]/metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')


batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = load_data_voc(batch_size, crop_size)

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 20, 0.001, 1e-3, try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)#(1, 3, 320, 480)
    pred = net(X.to(devices[0])).argmax(dim=1)#(1, 21, H/32, W/32) -> (1, H/32, W/32)
    return pred.reshape(pred.shape[1], pred.shape[2])#(1, H/32, W/32) -> (H/32, W/32)

def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=devices[0])#(21, 3)
    X = pred.long()#(H/32, W/32)
    return colormap[X, :]#(21, 3)[(H/32, W/32)] -> (H/32, W/32, 3)

test_images, test_labels = read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [
        X.permute(1, 2, 0),
        pred.cpu(),
        torchvision.transforms.functional.crop(
            test_labels[i], *crop_rect).permute(1, 2, 0)
    ]

# ========== 替换 d2l.show_images 开始 ==========
display_imgs = imgs[::3] + imgs[1::3] + imgs[2::3]

fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 3 * 2.5))
for i, ax in enumerate(axes.flat):
    img = display_imgs[i]
    
    # 统一转成 numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # 原图经过 Normalize，值可能不在 [0,1]，clamp 一下防止显示异常
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img, 0, 1)
    
    ax.imshow(img)
    ax.axis('off')

# 给每行加标题
titles = ['Input Image', 'Predicted', 'True Label']
for row, title in enumerate(titles):
    axes[row, 0].set_ylabel(title, fontsize=12, rotation=0, labelpad=60, va='center')

plt.tight_layout()
plt.show()