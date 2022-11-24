# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def tail(func):

    def wrapper(self, x):

        x_shape = x.shape

        x = x.view((-1,) + x_shape[-3:])
        x = func(self, x)
        x = x.view(x_shape[:-3] + x.shape[-3:])

        return x

    return wrapper







class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):

    def __init__(self, in_channels=256, out_channels=8, dropout=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channels, out_channels)
        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = Identity()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv3x3(self.in_channels, self.out_channels, stride)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        if downsample is None:
            self.downsample = Identity()
        else:
            self.downsample = downsample

    @tail
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out


class GumbelSoftmax(nn.Module):

    def __init__(self, interval=100, temperature=1.0):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)
        self.anneal_rate = 0.00003
        self.interval = 100
        self.counter = 0
        self.temperature_min = 0.5

    def anneal(self):
        self.temperature = max(
            self.temperature * torch.exp(-self.anneal_rate * self.counter),
            self.temperature_min)

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits):
        y = logits + self.sample_gumbel(logits)
        return self.softmax(y / self.temperature)

    def forward(self, logits):
        self.counter += 1
        if self.counter % self.interval == 0:
            self.anneal()
        y = self.gumbel_softmax_sample(logits)
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = (y_hard - y).detach() + y
        return y_hard
