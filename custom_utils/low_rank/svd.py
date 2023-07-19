import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils.extmath import randomized_svd
import os


class SVDWrapper(nn.Module):
    def __init__(self, layer, k, checkpoint_path, is_conv1d: bool = False):
        super().__init__()
        self.layer = layer
        self.k = k
        self.accelerator_checkpoint_path = os.path.join(checkpoint_path, "svd", f"k_{k}")
        self.is_conv1d = is_conv1d

        # Apply SVD to the layer's weight matrix
        # Apply SVD to the layer's weight matrix
        if os.path.exists(self.accelerator_checkpoint_path):
            accelerator_checkpoint = torch.load(self.accelerator_checkpoint_path)
            self.U = nn.Parameter(accelerator_checkpoint["U"])
            self.S = nn.Parameter(accelerator_checkpoint["S"])
            self.V_T = nn.Parameter(accelerator_checkpoint["V_T"])
        else:
            U, S, V_T = torch.linalg.svd(layer.weight, full_matrices=False)
            U = U[:, :k]
            S = S[:k]
            V_T = V_T[:k, :]
            self.U = nn.Parameter(U)
            self.S = nn.Parameter(S)
            self.V_T = nn.Parameter(V_T)
            os.makedirs(os.path.dirname(self.accelerator_checkpoint_path), exist_ok=True)
            torch.save(
                {"U": self.U.data, "S": self.S.data, "V_T": self.V_T.data}, self.accelerator_checkpoint_path)
        self.bias = self.layer.bias
        self.SU = nn.Parameter(self.S[None, :] * self.U)
        del self.layer

    def forward(self, x):
        if self.is_conv1d:
            size_out = x.size()[:-1] + (self.V_T.size()[-1],)
            xU = torch.mm(x.view(-1, x.size(-1)), self.U)
            xUSV_T = torch.addmm(self.layer.bias, xU, self.S[:, None] * self.V_T)
            xUSV_T = xUSV_T.view(*size_out)
            return xUSV_T
        else:
            xV = F.linear(x, self.V_T)
            xVSU_T = F.linear(xV, self.SU, self.bias)
            return xVSU_T

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict[prefix + 'U'] = self.U.detach() if not keep_vars else self.U
        state_dict[prefix + 'S'] = self.S.detach() if not keep_vars else self.S
        state_dict[prefix + 'V_T'] = self.V_T.detach() if not keep_vars else self.V_T
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.U.data.copy_(state_dict['U'])
        self.S.data.copy_(state_dict['S'])
        self.V_T.data.copy_(state_dict['V_T'])
        del state_dict['U']
        del state_dict['S']
        del state_dict['V_T']
        super().load_state_dict(state_dict, strict)