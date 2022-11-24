from blocks import *


class CenterContrastModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CenterContrastModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):

        x_center = torch.sum(x, dim=1) # N, C, H, W, take sum of matrices
        y = y.unsqueeze(2).unsqueeze(2).expand((-1, -1) + x_center.size()[-2:])

        x_center = torch.cat((x_center, y), dim = 1) # conditional on context inference
        x_center = self.bn(self.conv(x_center)) # compute a "center" of all matrices
        x = x - x_center.unsqueeze(1) # contrast the matrices with the "center"
        return x

