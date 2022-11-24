# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from blocks import *


class ContextInference(nn.Module):

    def __init__(self, num_attr, num_rule, gumbel = False, output_head_num = 1):
        """
        The output of ContextInference is used as input to multiple modules (e.g., contrast modules)
        """

        super(ContextInference, self).__init__()

        self.num_attr = num_attr
        self.num_rule = num_rule
        self.gumbel = gumbel
        self.output_head_num = output_head_num

        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.lin_attr_rule = nn.Linear(64, self.num_attr * self.num_rule)
        if self.gumbel:
            self.softmax = GumbelSoftmax(temperature=0.5)
        else:
            self.softmax = nn.Softmax(dim=-1)
        self.lin_bottleneck = nn.Linear(self.num_rule, 64, bias = False)

        self.output_heads = []
        for ii in range(self.output_head_num):
            self.output_heads.append(MLP(in_dim=64, out_dim=64))

    def forward(self, context):

        # context: N, C, H, W
        context = self.avgpool2d(context).squeeze() # N, C

        # PRETEND to predict the probability that,for each given attribute, a particular rule is applied on it.
        pred_attr_rule = self.lin_attr_rule(context) # N, num_attr * num_rule
        pred_attr_rule = pred_attr_rule.view(-1, self.num_attr, self.num_rule) # N, num_attr, num_rule
        pred_attr_rule = self.softmax(pred_attr_rule) # N, num_attr, num_rule

        # from a bottleneck to multiple heads
        pred_attr = self.lin_bottleneck(pred_attr_rule) # N, num_attr, 64
        pred = torch.sum(pred_attr, dim = 1) # N, 64
        outputs = []
        for head in self.output_heads:
            outputs.append(head(pred))

        return outputs


class EntryEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EntryEncoder, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        # batch, entries, height, width, our images has only a single channel.
        N, E, H, W = x.shape

        x = x.view(-1, 1, H, W)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view((N, E) + x.size()[-3])

        return x


class SequenceEncoder(nn.Module):
    """
    This is only very specific method in CoPINet.
    It might not be a reasonable one. And there has to be better ones.
    In fact, it looks incorrect because it sums up features of entries in a sequence.
    Maybe, structures like RN or SCL would works better here.
    Also, CoPINet uses different sequence encoders for rows, columns, context, and inference,
    which seems wrong.
    """

    def __init__(self, in_channels, out_channels):
        super(SequenceEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn_row = nn.BatchNorm2d(64, 64)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):

        # batch, sequences, entries, channels, height, width
        # N, S, E, C, H, W = x.shape

        x = torch.sum(x, dim = 2)
        x = self.conv(x)
        x = self.bn_row(x)
        x = self.relu(x)

        return x


class CenterContrastModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CenterContrastModule, self).__init__()

        self.conv = conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):

        x_center = torch.sum(x, dim=1) # N, C, H, W, take sum of matrices
        y = y.unsqueeze(2).unsqueeze(2).expand((-1, -1) + x_center.size()[-2:])

        x_center = torch.cat((x_center, y), dim = 1) # conditional on context inference
        x_center = self.bn(self.conv(x_center)) # compute a "center" of all matrices
        x = x - x_center.unsqueeze(1) # contrast the matrices with the "center"
        return x


class CoPINet(nn.Module):

    def __init__(self, num_attr=10, num_rule=6, gumbel=False, dropout=False):
        super(CoPINet, self).__init__()
        self.num_attr = num_attr
        self.num_rule = num_rule
        self.gumbel = gumbel

        self.entryEcnoder = EntryEncoder(1, 64)
        self.sequenceEncoder = SequenceEncoder(64, 64)
        self.contextInference = ContextInference(self.num_attr, self.num_rule, self.gumbel, output_head_num = 2)

        self.contrast1 = CenterContrastModule(64 + 64, 64)
        self.resBlock1 = ResBlock(64, 128, stride=2, downsample=nn.Sequential(conv1x1(64, 128, stride=2), nn.BatchNorm2d(128)))

        self.contrast2 = CenterContrastModule(128 + 64, 128)
        self.resBlock2 = ResBlock(128, 256, stride = 2, downsample = nn.Sequential(conv1x1(128, 256, stride = 2), nn.BatchNorm2d(256)))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MLP(in_dim=256, out_dim=1, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        N, E, H, W = x.shape # (Batch, Matrix and Choices Entries, H, W)

        entries = self.entryEcnoder(x) # N, E, C, H, W

        entry_ids_for_rows = [[0, 1, 2],
                              [3, 4, 5],
                              [6, 7, 8],
                              [6, 7, 9],
                              [6, 7, 10],
                              [6, 7, 11],
                              [6, 7, 12],
                              [6, 7, 13],
                              [6, 7, 14],
                              [6, 7, 15]]
        rows = entries[:, entry_ids_for_rows, :, :, :]
        rows = self.sequenceEncoder(rows) # N, S, C, H, W

        entry_ids_for_cols = [[0, 3, 6],
                              [1, 4, 7],
                              [2, 5, 8],
                              [2, 5, 9],
                              [2, 5, 10],
                              [2, 5, 11],
                              [2, 5, 12],
                              [2, 5, 13],
                              [2, 5, 14],
                              [2, 5, 15]]
        cols = entries[:, entry_ids_for_cols, :, :, :]
        cols = self.sequenceEncoder(cols) # N, S, C, H, W

        # aggregate row and col features
        context_rows = rows[: , 0] + rows[: , 1]
        context_cols = cols[: , 0] + cols[: , 1]
        context = context_rows + context_cols
        contrast1_param, contrast2_param = self.contextInference(context)

        # Choices Inference
        row_ids_for_matrices = [[0, 1, 2],
                                [0, 1, 3],
                                [0, 1, 4],
                                [0, 1, 5],
                                [0, 1, 6],
                                [0, 1, 7],
                                [0, 1, 8],
                                [0, 1, 9]]
        matrices_by_rows = torch.sum(rows[:, row_ids_for_matrices, :, :, :], dim = 2)
        col_ids_for_matrices = row_ids_for_matrices
        matrices_by_cols = torch.sum(cols[:, col_ids_for_matrices, :, :, :], dim = 2)
        matrices = matrices_by_rows + matrices_by_cols # N, choices, C, H, W

        matrices = self.contrast1(matrices, contrast1_param)

        matrices = matrices.view((-1,) + matrices.size()[-3]) # N * choices, C, H, W
        matrices = self.resBlock1(matrices) # N * choices, C, H, W
        matrices = matrices.view((N, 8) + matrices.size()[-3]) # N, choices, C, H, W

        matrices = self.contrast2(matrices, contrast2_param)

        matrices = matrices.view((-1,) + matrices.size()[-3]) # N * choices, C, H, W
        matrices = self.resBlock2(matrices) # N * choices, C, H, W

        matrices = self.avgpool(matrices).squeeze() # N * choices, C
        matrices = self.mlp(matrices) # N * choices
        pred = matrices.view(-1, 8) # N, choices

        return pred
