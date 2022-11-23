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

        self.res1_contrast = conv3x3(64 + 64, 64)
        self.res1_contrast_bn = nn.BatchNorm2d(64)
        self.res1 = ResBlock(64, 128, stride=2, downsample=nn.Sequential(conv1x1(64, 128, stride=2), nn.BatchNorm2d(128)))

        self.res2_contrast = conv3x3(128 + 64, 128)
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
        # (Batch_size, Matrix Entries + Answer Choices, Height, Width)
        N, E, H, W = x.shape

        # Inference Branch
        context = x[:, :8, :, :]  # 3x3 matrix with the last entry missing
        contrast1_bias, contrast2_bias = self.contextInference(context)







        #TODO find a place to do the following without specifying 20 and 10
        contrast1_bias = contrast1_bias.view(-1, 64, 1, 1).expand(-1, -1, 20, 20)
        contrast2_bias = contrast2_bias.view(-1, 64, 1, 1).expand(-1, -1, 10, 10)







        # Perception Branch
        input_features = self.maxpool(self.relu(self.bn1(self.conv1(x.view(-1, 80, 80).unsqueeze(1)))))
        input_features = input_features.view(-1, 16, 64, 20, 20)

        choices_features = input_features[:, 8:, :, :, :].unsqueeze(2)  # N, 8, 64, 20, 20 -> N, 8, 1, 64, 20, 20

        row1_features = torch.sum(input_features[:, 0:3, :, :, :], dim=1)  # N, 64, 20, 20
        row2_features = torch.sum(input_features[:, 3:6, :, :, :], dim=1)  # N, 64, 20, 20
        row3_pre = input_features[:, 6:8, :, :, :].unsqueeze(1).expand(N, 8, 2, 64, 20, 20)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        row3_features = torch.sum(torch.cat((row3_pre, choices_features), dim=2), dim=2).view(-1, 64, 20, 20)  # N, 8, 3, 64, 20, 20 -> N, 8, 64, 20, 20 -> N * 8, 64, 20, 20
        row_features = self.relu(self.bn_row(self.conv_row(torch.cat((row1_features, row2_features, row3_features), dim=0))))

        row1 = row_features[:N, :, :, :].unsqueeze(1).unsqueeze(1).expand(N, 8, 1, 64, 20, 20)
        row2 = row_features[N:2 * N, :, :, :].unsqueeze(1).unsqueeze(1).expand(N, 8, 1, 64, 20, 20)
        row3 = row_features[2 * N:, :, :, :].view(-1, 8, 64, 20,20).unsqueeze(2)
        final_row_features = torch.sum(torch.cat((row1, row2, row3), dim=2), dim=2)

        col1_features = torch.sum(input_features[:, 0:9:3, :, :, :], dim=1)  # N, 64, 20, 20
        col2_features = torch.sum(input_features[:, 1:9:3, :, :, :], dim=1)  # N, 64, 20, 20
        col3_pre = input_features[:, 2:8:3, :, :, :].unsqueeze(1).expand(N, 8, 2, 64, 20, 20)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        col3_features = torch.sum(torch.cat((col3_pre, choices_features), dim=2), dim=2).view(-1, 64, 20, 20)  # N, 8, 3, 64, 20, 20 -> N, 8, 64, 20, 20 -> N * 8, 64, 20, 20
        col_features = self.relu(self.bn_col(self.conv_col(torch.cat((col1_features, col2_features, col3_features), dim=0))))

        col1 = col_features[:N, :, :, :].unsqueeze(1).unsqueeze(1).expand(N, 8, 1, 64, 20, 20)
        col2 = col_features[N:2 * N, :, :, :].unsqueeze(1).unsqueeze(1).expand(N, 8, 1, 64, 20, 20)
        col3 = col_features[2 * N:, :, :, :].view(-1, 8, 64, 20, 20).unsqueeze(2)
        final_col_features = torch.sum(torch.cat((col1, col2, col3), dim=2), dim=2)

        input_features = final_row_features + final_col_features
        input_features = input_features.view(-1, 64, 20, 20)

        res1_in = input_features.view(-1, 8, 64, 20, 20)
        res1_contrast = self.res1_contrast_bn(self.res1_contrast(torch.cat((torch.sum(res1_in, dim=1), contrast1_bias), dim=1)))
        res1_in = res1_in - res1_contrast.unsqueeze(1)
        res2_in = self.res1(res1_in.view(-1, 64, 20, 20))
        res2_in = res2_in.view(-1, 8, 128, 10, 10)
        res2_contrast = self.res2_contrast_bn(self.res2_contrast(torch.cat((torch.sum(res2_in, dim=1), contrast2_bias), dim=1)))
        res2_in = res2_in - res2_contrast.unsqueeze(1)
        out = self.res2(res2_in.view(-1, 128, 10, 10))

        avgpool = self.avgpool(out)
        avgpool = avgpool.view(-1, 256)
        final = avgpool
        final = self.mlp(final)
        return final.view(-1, 8)
