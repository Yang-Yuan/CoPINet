# -*- coding: utf-8 -*-

from blocks import *
from const import *
from EntryEncoder import EntryEncoder
from SequenceEncoder import SequenceEncoder
from ContextInference import ContextInference
from CenterContrastModule import CenterContrastModule


class CoPINet(nn.Module):

    def __init__(self, num_attr=10, num_rule=6, gumbel=False, dropout=False):
        super(CoPINet, self).__init__()
        self.num_attr = num_attr
        self.num_rule = num_rule
        self.gumbel = gumbel

        self.ee = EntryEncoder(in_channels = 1, out_channels = 64)
        self.se = SequenceEncoder(in_channels = self.ee.out_channels, out_channels = 64)

        #TODO it lacks a matrix encoder

        self.ci = ContextInference(in_channels = self.se.out_channels, out_channels = 64,
                                   num_attr = self.num_attr, num_rule = self.num_rule,
                                   gumbel = self.gumbel, output_head_num = 2)

        self.con1 = CenterContrastModule(in_channels = self.se.out_channels + self.ci.out_channels, out_channels = 64)
        self.res1 = ResBlock(in_channels = self.con1.out_channels, out_channels = 128, stride=2,
                             downsample=nn.Sequential(conv1x1(64, 128, stride=2), nn.BatchNorm2d(128)))

        self.con2 = CenterContrastModule(in_channels = self.res1.out_channels + self.ci.out_channels, out_channels = 128)
        self.res2 = ResBlock(in_channels = self.con2.out_channels, out_channels = 256,
                             stride = 2, downsample = nn.Sequential(conv1x1(128, 256, stride = 2), nn.BatchNorm2d(256)))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MLP(in_channels = self.res2.out_channels, out_channels =1, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        N, E, H, W = x.shape # (Batch, Matrix and Choices Entries, H, W)

        x = x.unsqueeze(2) # add the channel dimension
        entries = self.ee(x) # N, E, C, H, W

        rows = entries[:, ENTRY_IDS_FOR_ROWS, :, :, :] # N, S, E, C, H, W
        rows = self.se(rows) # N, S, C, H, W

        cols = entries[:, ENTRY_IDS_FOR_COLS, :, :, :] # N, S, E, C, H, W
        cols = self.se(cols) # N, S, C, H, W

        # Context Inference based on first two rows and columns of the matrix
        context_rows = rows[: , 0] + rows[: , 1]
        context_cols = cols[: , 0] + cols[: , 1]
        context = context_rows + context_cols
        contrast1_param, contrast2_param = self.ci(context)

        # Choices Inference
        matrices_by_rows = torch.sum(rows[:, ROW_IDS_FOR_MATRICES, :, :, :], dim = 2)
        matrices_by_cols = torch.sum(cols[:, COL_IDS_FOR_MATRICES, :, :, :], dim = 2)
        matrices = matrices_by_rows + matrices_by_cols # N, choices, C, H, W

        matrices = self.con1(matrices, contrast1_param)

        matrices = self.res1(matrices) # N, choices, C, H, W

        matrices = self.con2(matrices, contrast2_param)

        matrices = self.res2(matrices) # N, choices, C, H, W

        matrices = self.avgpool(matrices).squeeze() # N, choices, C
        matrices = self.mlp(matrices) # N, choices, 1
        pred = matrices.squeeze() # N, choices

        return pred
