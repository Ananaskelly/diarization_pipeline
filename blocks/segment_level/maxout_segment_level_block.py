import torch
import torch.nn as nn


class MaxoutSegmentLevelBlock(nn.Module):

    def __init__(self, input_dim, output_dim, batch_norm=True, activation='mfm'):

        super(MaxoutSegmentLevelBlock, self).__init__()

        self.num_layers = len(input_dim)
        self.batch_norm = batch_norm
        self.mfm = nn.ModuleList([])
        self.bn = nn.ModuleList([])

        for idx in range(self.num_layers):
            if activation == 'mfm':
                self.mfm.append(MaxoutLinear(input_dim[idx], output_dim[idx]))
            else:
                self.mfm.append(nn.Linear(input_dim[idx], output_dim[idx]))

            if self.batch_norm:
                self.bn.append(nn.BatchNorm1d(output_dim[idx], affine=False))

    def forward(self, x):

        for idx in range(self.num_layers):
            x = self.mfm[idx](x)

            if self.batch_norm:
                x = self.bn[idx](x)

        return x


class MaxoutLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return torch.max(self.linear1(x), self.linear2(x))
