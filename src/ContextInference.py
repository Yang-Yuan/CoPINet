from torch import nn
import torch
from blocks import GumbelSoftmax, MLP


class ContextInference(nn.Module):

    def __init__(self, in_channels, out_channels, num_attr, num_rule, gumbel = False, output_head_num = 1):
        """
        The output of ContextInference is used as input to multiple modules (e.g., contrast modules)
        """

        super(ContextInference, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attr = num_attr
        self.num_rule = num_rule
        self.gumbel = gumbel
        self.output_head_num = output_head_num

        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.lin_attr_rule = nn.Linear(in_channels, self.num_attr * self.num_rule)
        if self.gumbel:
            self.softmax = GumbelSoftmax(temperature=0.5)
        else:
            self.softmax = nn.Softmax(dim=-1)
        self.lin_bottleneck = nn.Linear(self.num_rule, 64, bias = False)

        self.output_heads = nn.ModuleList()
        for ii in range(self.output_head_num):
            self.output_heads.append(MLP(in_channels = 64, out_channels = out_channels))

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

