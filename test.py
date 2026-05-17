
import torch
from torch import nn

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
print(f'x',X)
print(f'k',K)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2,padding=2, stride=3,bias=False)
tconv.weight.data = K
print(f'tconv',tconv(X))