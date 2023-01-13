import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


class AMSoftmax(nn.Module):
    def __init__(self, emb_size, num_classes, margin=0.3, scale=15, **kwargs):
        super(AMSoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = emb_size
        self.num_classes = num_classes
        self.W = torch.nn.Parameter(torch.randn(emb_size, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label):

        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)

        cos_th = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        label_view = label_view.cpu()
        delt_cos_th = torch.zeros(cos_th.size()).scatter_(1, label_view, self.m)
        delt_cos_th = delt_cos_th.to(x.device)
        cos_th_m = cos_th - delt_cos_th
        cos_th_m_s = self.s * cos_th_m
        return cos_th_m_s
