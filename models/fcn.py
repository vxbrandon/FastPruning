import torch
import torch.nn as nn

from typing import Callable, Union, List
from collections import OrderedDict


class FCN(nn.Module):
    def __init__(
        self,
        input_dims: int = 784,
        num_classes: int = 10,
        num_layers: int = 5,
        hidden_dims: Union[int, List] = 784,
        is_softmax: bool = False,
        act_func: Callable = nn.ReLU,
    ):
        super(FCN, self).__init__()
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.act_func = act_func
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims for _ in range(num_layers - 1)]
            layer_dims = [input_dims] + hidden_dims + [num_classes]
        else:
            assert len(hidden_dims) == num_layers - 1
            layer_dims = [input_dims] + hidden_dims + [num_classes]
        print("FCN architecture: ", layer_dims)

        modules = OrderedDict()
        for idx in range(0, num_layers - 1):
            modules[f"fc{idx}"] = nn.Linear(layer_dims[idx], layer_dims[idx + 1])
            modules[f"act{idx}"] = self.act_func()
        modules[f"fc{num_layers - 1}"] = nn.Linear(layer_dims[num_layers - 1], num_classes)

        self.fcn = nn.Sequential(modules)

    def forward(self, x):
        return self.fcn(x)


def exp():
    fcn = FCN()
    input = torch.rand(784)

    print(fcn(input))


if __name__ == "__main__":
    exp()
