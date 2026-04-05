
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print(net.state_dict())
torch.save(net.state_dict(), 'mlp_params.pth')

clone = MLP()
clone.load_state_dict(torch.load('mlp_params.pth'))
print(Y == clone(X))
print(f'GPU数量: {torch.cuda.device_count()}')