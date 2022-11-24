from torch import nn
import torch
from blocks import conv3x3, tail


class SequenceEncoder(nn.Module):
    """
    This is only very specific method in CoPINet.
    It might not be a reasonable one. And there has to be better ones.
    In fact, it looks incorrect because it sums up features of entries in a sequence.
    Maybe, structures like RN or SCL would works better here.
    Also, original CoPINet uses different sequence encoders for rows, columns, context, and inference,
    which seems wrong.
    """

    def __init__(self, in_channels, out_channels):
        super(SequenceEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv3x3(self.in_channels, self.out_channels)
        self.bn_row = nn.BatchNorm2d(64, 64)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):

        # batch, sequences, entries, channels, height, width
        # N, S, E, C, H, W = x.shape

        # aggregation, could better
        x = torch.sum(x, dim = 2)

        x = self.forward_(x)

        return x

    @tail
    def forward_(self, x):
        x = self.conv(x)
        x = self.bn_row(x)
        x = self.relu(x)
        return x
