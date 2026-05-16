import os
import torch
import torchvision
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np

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

train_features, train_labels = read_voc_images(voc_dir, True)

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

