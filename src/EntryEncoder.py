from torch import nn
from blocks import tail


class EntryEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EntryEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @tail
    def forward(self, x):

        # batch, entries, height, width, our images has only a single channel.
        # N, E, H, W = x.shape

        # x = x.view(-1, 1, H, W)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = x.view((N, E) + x.size()[-3 :])

        return x
