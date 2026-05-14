import os
import hashlib
import urllib.request
import zipfile
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==================== 数据下载工具 ====================
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB = {
    'banana-detection': (
        DATA_URL + 'banana-detection.zip',
        '5de26c8fce5ccdea9f91267273464dc968d20d72'
    )
}

def download(name, cache_dir=os.path.join('..', 'data')):
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'Downloading {fname} from {url}...')
    urllib.request.urlretrieve(url, fname)
    sha1 = hashlib.sha1()
    with open(fname, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    assert sha1.hexdigest() == sha1_hash, f'File corrupted: {fname}'
    return fname

def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
    else:
        raise ValueError(f'Only zip files can be extracted, got {ext}')
    return data_dir

# ==================== 数据集加载 ====================
def read_data_bananas(is_train=True):
    data_dir = download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

# ==================== 可视化工具（替代 d2l） ====================
def show_images(imgs, num_rows, num_cols, scale=2):
    """显示图像网格，替代 d2l.show_images"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axis('off')
    return axes

def show_bboxes(ax, bboxes, colors=None):
    """在图像上绘制边界框，替代 d2l.show_bboxes"""
    if colors is None:
        colors = ['w'] * len(bboxes)
    for bbox, color in zip(bboxes, colors):
        x1, y1, x2, y2 = [float(c) for c in bbox]
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height,
                        fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

# ==================== 主程序 ====================
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
print(batch[0].shape, batch[1].shape)

# 可视化前 10 张图像及其标注框
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])

plt.show()