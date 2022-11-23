# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from blocks import *



class ContextInference(nn.Module):

    def __init__(self, num_attr, num_rule, gumbel = False, output_head_num = 1):
        super(ContextInference, self).__init__()

        self.num_attr = num_attr
        self.num_rule = num_rule
        self.gumbel = gumbel
        self.output_head_num = output_head_num

        # for processing each entry image
        self.conv_entry = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_entry = nn.BatchNorm2d(64)
        self.max_pool_entry = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # for processing rows
        self.conv_row = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn_row = nn.BatchNorm2d(64, 64)

        # for processing columns
        self.conv_col = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn_col = nn.BatchNorm2d(64, 64)

        # for processing the final aggregated feature
        self.lin_attr_rule = nn.Linear(64, self.num_attr * self.num_rule)

        if self.gumbel:
            self.softmax = GumbelSoftmax(temperature=0.5)
        else:
            self.softmax = nn.Softmax(dim=-1)

        # the output of this ContextInference is used as
        # part of input to the modules (e.g., contrast modules)
        self.output_heads = []
        self.bottleneck = nn.Linear(self.num_rule, 64, bias=False)
        for ii in range(self.output_head_num):
            self.output_heads.append(MLP(in_dim=64, out_dim=64))


    def forward(self, context_entries):

        # (Batch_size, Context Matrix Entries, Height, Width)
        N, E, H, W = context_entries.shape

        # encode each entry in all matrices in the batch
        context = context_entries.view(-1, H, W).unsqueeze(1)
        context = self.conv_entry(context)
        context = self.bn_entry(context)
        context = self.relu(context)
        context = self.max_pool_entry(context)

        # restore shape from (N*E, channel, new_heighet, new_width)
        # to (N, E, channel, new_heighet, new_width)
        batch_entry_shape = (N, E) + context.size()[-3:]
        context = context.view(batch_entry_shape)

        # aggregate entry features to a feature for the first two rows
        row1 = torch.sum(context[:, (0, 1, 2), :, :, :], dim = 1)
        row1 = self.conv_row(row1)
        row1 = self.bn_row(row1)
        row1 = self.relu(row1)
        row2 = torch.sum(context[:, (3, 4, 5), :, :, :], dim = 1)
        row2 = self.conv_row(row2)
        row2 = self.bn_row(row2)
        row2 = self.relu(row2)
        row = row1 + row2

        # aggregate entry features to a feature for the first two columns
        col1 = torch.sum(context[:, (0, 3, 6), :, :, :], dim = 1)
        col1 = self.conv_col(col1)
        col1 = self.bn_col(col1)
        col1 = self.relu(col1)
        col2 = torch.sum(context[:, (1, 4, 7), :, :, :], dim = 1)
        col2 = self.conv_col(col2)
        col2 = self.bn_col(col2)
        col2 = self.relu(col2)
        col = col1 + col2

        # aggregate row and col features
        context = row + col
        context = self.avgpool(context).squeeze()

        # PRETEND!!! to predict the probability that,
        # for given each attribute, a particular rule is applied on it.
        pred_attr_rule = self.lin_attr_rule(context)
        pred_attr_rule = pred_attr_rule.view(-1, self.num_rule)
        pred_attr_rule = self.softmax(pred_attr_rule)

        # from a bottleneck to multiple heads
        pred_attr = self.bottleneck(pred_attr_rule)
        pred_attr = pred_attr.view(-1, self.num_attr, pred_attr.size([-1]))
        pred = torch.sum(pred_attr, dim = 1)
        outputs = []
        for head in self.output_heads:
            outputs.append(head(pred))

        return outputs










class CoPINet(nn.Module):

    def __init__(self, num_attr=10, num_rule=6, gumbel=False, dropout=False):
        super(CoPINet, self).__init__()
        self.num_attr = num_attr
        self.num_rule = num_rule
        self.gumbel = gumbel

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.contextInference = ContextInference(self.num_attr, self.num_rule, self.gumbel, output_head_num = 2)

        self.res1_contrast_conv = conv3x3(64 + 64, 64)
        self.res1_contrast_bn = nn.BatchNorm2d(64)
        self.res1 = ResBlock(64, 128, stride=2, downsample=nn.Sequential(conv1x1(64, 128, stride=2), nn.BatchNorm2d(128)))

        self.res2_contrast_conv = conv3x3(128 + 64, 128)
        self.res2_contrast_bn = nn.BatchNorm2d(128)
        self.res2 = ResBlock(128, 256, stride=2, downsample=nn.Sequential(conv1x1(128, 256, stride=2), nn.BatchNorm2d(256)))

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
        # (N, Matrix Entries + Answer Choices, H, W)
        N, E, H, W = x.shape

        # Context Inference Branch
        context = x[:, :8, :, :]  # 3x3 matrix with the last entry missing
        contrast1_bias, contrast2_bias = self.contextInference(context)


        #TODO find a place to do the following without specifying 20 and 10
        contrast1_bias = contrast1_bias.view(-1, 64, 1, 1).expand(-1, -1, 20, 20)
        contrast2_bias = contrast2_bias.view(-1, 64, 1, 1).expand(-1, -1, 10, 10)

        #TODO rewrite this to use a single entry encoder for all branches.

        # Main Inference

        # encode each entry in all matrices in the batch
        entries = x.view((-1,) + x.size()[-2]).unsqueeze(1) # N * E, 1, H, W
        entries = self.conv1(entries)
        entries = self.bn1(entries)
        entries = self.relu(entries)
        entries = self.maxpool(entries) # N * E, C, H, W
        batch_entry_shape = (N, E) + entries.size()[-3:]
        entries = entries.view(batch_entry_shape) # N, E, C, H, W

        choices = entries[:, 8:, :, :, :] # N, choices, C, H, W
        choices = choices.unsqueeze(2) # N, choices, 1, C, H, W, expand to match copies of row3

        # aggregate entries features to a feature for all (2+8) rows
        row1 = torch.sum(entries[:, (0, 1, 2), :, :, :], dim=1)  # N, C, H, W
        row2 = torch.sum(entries[:, (3, 4, 5), :, :, :], dim=1)  # N, C, H, W
        row3 = entries[:, (6, 7), :, :, :]  # N, 2, C, H, W
        row3 = row3.unsqueeze(1).expand(-1, 8, -1, -1, -1, -1) # N, 8, 2, C, H, W, expand to match 8 choices
        row3 = torch.cat((row3, choices), dim=2) # N, choices, 3, C, H, W, complete row3 with each choice
        row3 = torch.sum(row3, dim=2) # N, choices, C, H, W, take sum over the entry dim as row1 and row2
        row3 = row3.view((-1,) + row3.size()[-3])  # N * choices, C, H, W
        row = torch.cat((row1, row2, row3), dim=0) # N + N + N * choices, C, H, W
        row = self.relu(self.bn_row(self.conv_row(row))) # N + N + B * choices, C, H, W
        row1 = row[: N, :, :, :].unsqueeze(1).unsqueeze(1).expand(-1, 8, -1, -1, -1, -1) # N, choices, 1, C, H, W, expand to match choices
        row2 = row[N : 2 * N, :, :, :].unsqueeze(1).unsqueeze(1).expand(-1, 8, -1, -1, -1, -1) # N, choices, 1, C, H, W, expand to match choices
        row3 = row[2 * N: , :, :, :].view((N, 8) + row.size()[-3]).unsqueeze(2) # N, choices, 1, C, H, W
        row = torch.cat((row1, row2, row3), dim=2) # N, choices, 3, C, H, W
        row = torch.sum(row, dim=2) # N, choices, C, H, W, sum all rows

        # aggregate entries features to a feature for all (2+8) rows
        col1 = torch.sum(entries[:, (0, 3, 6), :, :, :], dim=1)  # N, C, H, W
        col2 = torch.sum(entries[:, (1, 4, 7), :, :, :], dim=1)  # N, C, H, W
        col3 = entries[:, (2, 5), :, :, :] # N, 2, C, H, W
        col3 = col3.unsqueeze(1).expand(-1, 8, -1, -1, -1, -1)  # N, choices, 2, C, H, W, expand to match 8 choices
        col3 = torch.cat((col3, choices), dim=2) # N, choices, 3, C, H, W, complete each col with each choice
        col3 = torch.sum(col3, dim=2) # N, choices, C, H, W, take sum over the entry dim as col1 and col3
        col3 = col3.view((-1,) + row3.size()[-3])  # N * choices, C, H, W
        col = torch.cat((col1, col2, col3), dim=0) # N + N + N * choices, C, H, W
        col = self.relu(self.bn_col(self.conv_col(col))) # N + N + N * choices, C, H, W
        col1 = col[: N, :, :, :].unsqueeze(1).unsqueeze(1).expand(-1, 8, -1, -1, -1, -1) # N, choices, 1, C, H, W, expand to match choices
        col2 = col[N : 2 * N, :, :, :].unsqueeze(1).unsqueeze(1).expand(-1, 8, -1, -1, -1, -1) # N, choices, 1, C, H, W, expand to match choices
        col3 = col[2 * N: , :, :, :].view(-1, 8, 64, 20, 20).unsqueeze(2) # N, choices, 1, C, H, W
        col = torch.cat((col1, col2, col3), dim=2) # N, choices, 3, C, H, W
        col = torch.sum(col, dim=2) # N, choices, C, H, W, sum all columns

        # combined to get a feature for the entire matrix completed by each answer choices
        matrices = row + col # N, choices, C, H, W
        matrix_new = matrices.view((-1,) + matrices.size()[-3]) # N * choices, C, H, W

        # contrast module 1
        matrices_center = torch.sum(matrices, dim=1) # N, C, H, W, take sum of matrices
        matrices_center = torch.cat((matrices_center, contrast1_bias), dim = 1) # conditional on context inference
        matrices_center = self.res1_contrast_bn(self.res1_contrast_conv(matrices_center)) # compute a "center" of all matrices
        matrices = matrices - matrices_center.unsqueeze(1) # contrast the matrices with the "center"

        # residual block 1
        matrices = matrices.view((-1,) + matrices.size()[-3]) # N * choices, C, H, W
        matrices = self.res1(matrices) # N * choices, C, H, W
        matrices = matrices.view((N, 8) + matrices.size()[-3]) # N, choices, C, H, W

        #contrast module 2
        matrices_center = torch.sum(matrices, dim=1) # B, C, H, W
        matrices_center = torch.cat((matrices_center, contrast2_bias), dim=1) # conditional on context inference
        matrices_center = self.res2_contrast_bn(self.res2_contrast_conv(matrices_center)) # compute a "center" of all matrices
        matrices = matrices - matrices_center.unsqueeze(1) # contrast the matrices with the "center"

        # residual block 2
        matrices = matrices.view((-1,) + matrices.size()[-3]) # N * choices, C, H, W
        out = self.res2(matrices) # N * choices, C, H, W

        avgpool = self.avgpool(out).squeeze() # N * choices, C
        final = self.mlp(avgpool) # N * choices
        return final.view(-1, 8) # N, choices
