import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        N, = x.size()
        return x.view(N, -1)
