import torch.nn as nn

from blocks.frame_level import build_resnet18, build_resnet50
from blocks.pooling_level import StatPoolLayer
from blocks.segment_level import MaxoutSegmentLevelBlock

from blocks.heads import AMSoftmax


class ResNetSiNET(nn.Module):

    def __init__(self, cfg, num_classes):
        super(ResNetSiNET, self).__init__()

        self.fl = build_resnet50(num_classes)
        self.sp = StatPoolLayer(mode=2)
        self.sl = MaxoutSegmentLevelBlock(input_dim=cfg.input_dim,
                                          output_dim=cfg.output_dim)

        self.head = AMSoftmax(emb_size=cfg.emb_size,
                              num_classes=num_classes)

    def forward(self, x, label, **kwargs):

        fl_out = self.fl(x)
        sp_out = self.sp(fl_out)
        sl_out = self.sl(sp_out)
        out = self.head(sl_out, label)

        return out
