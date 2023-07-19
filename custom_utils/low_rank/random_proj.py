import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils.extmath import randomized_svd
import os


class ProjWrapper(nn.Module):
    def __init__(self, layer, k, checkpoint_path, weight_proj: torch.tensor = None, input_proj: torch.tensor = None, is_conv1d: bool = False):
        super().__init__()
        self.layer = layer

        # Freeze Original Layer
        for param in layer.parameters():
            param.requires_grad = False

        self.weight_proj = nn.Parameter(torch.rand(k, layer.weight.size()[1]) if weight_proj is None else weight_proj)
        self.input_proj = nn.Parameter(torch.rand(k, layer.weight.size()[1]) if input_proj is None else input_proj)
        self.is_conv1d = is_conv1d


    def forward(self, x):
        if self.is_conv1d:
            size_out = x.size()[:-1] + (self.V_T.size()[-1],)
            xU = torch.mm(x.view(-1, x.size(-1)), self.U)
            xUSV_T = torch.addmm(self.layer.bias, xU, self.S[:, None] * self.V_T)
            xUSV_T = xUSV_T.view(*size_out)
            return xUSV_T
        else:
            downsample_weight = F.linear(self.layer.weight, self.weight_proj)
            downsample_input = F.linear(x, self.input_proj)
            return F.linear(downsample_input, downsample_weight, self.layer.bias)
