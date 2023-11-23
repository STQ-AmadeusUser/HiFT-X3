# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.backbone.alexnet_deploy import AlexNet
from pysot.models.utile_deploy.utile import HiFT


class ModelBuilder(nn.Module):
    def __init__(self, device='cpu'):
        super(ModelBuilder, self).__init__()

        self.backbone = AlexNet().to(device)
        self.grader = HiFT(cfg).to(device)

    def forward(self, template, search):
        """ only used in training
        """

        zf = self.backbone(template)  # bx384x10x10  bx384x8x8  bx256x6x6
        xf = self.backbone(search)  # bx384x30x30  bx384x28x28  bx256x26x26

        loc, cls1, cls2 = self.grader(xf, zf)

        # loc: 1x4x11x11
        # cls1: 1x2x11x11
        # cls2: 1x1x11x11

        # for the convenience of X3
        cls1 = cls1.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)  # 121x2
        cls1 = F.softmax(cls1, dim=1)[:, 1]  # (121,)
        cls2 = cls2.contiguous().view(-1)  # (121,)

        return loc, cls1, cls2

    def template(self, z):
        with t.no_grad():
            zf = self.backbone(z)
            self.zf = zf

    def track(self, x):
        with t.no_grad():
            xf = self.backbone(x)
            loc, cls1, cls2 = self.grader(xf, self.zf)
            return {
                'cls1': cls1,
                'cls2': cls2,
                'loc': loc
            }
