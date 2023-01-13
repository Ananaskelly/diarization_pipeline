import torch
import torch.nn as nn


class StatPoolLayer(nn.Module):

    def __init__(self, mode):

        super(StatPoolLayer, self).__init__()
        self.mode = mode

    def forward(self, x):
        dim = 3

        mean_x = x.mean(dim)
        mean_x2 = x.pow(2).mean(dim)

        std_x = nn.functional.relu(mean_x2 - mean_x.pow(2)).sqrt()

        if self.mode == 0:
            out = mean_x
        elif self.mode == 1:
            out = std_x
        elif self.mode == 2:
            out = torch.cat([mean_x, std_x], dim=-1)

        out = torch.flatten(out, 1)

        return out
