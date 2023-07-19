import torch
import torch.nn as nn

from typing import Callable, Union, Iterable


class FCN(nn.Module):
    def __init__(
        self,
        input_dims: int = 784,
        num_classes: int = 10,
        num_layers: int = 5,
        hidden_dims: Union[int, Iterable] = 784,
        is_softmax: bool = False,
        act_func: Callable = nn.ReLU,
    ):
        super(FCN, self).__init__()
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.act_func = act_func
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims for _ in range(num_layers-1)]

        modules = []

        # Initial Layer
        modules.append(nn.Linear(input_dims, hidden_dims[0]))
        modules.append(self.act_func())

        # Intermediate Layers
        for idx in range(1, num_layers - 1):
            modules.append(nn.Linear(hidden_dims[idx], hidden_dims[idx]))
            modules.append(self.act_func())
        # Final Layer
        modules.append(nn.Linear(hidden_dims[num_layers-2], num_classes))
        modules.append(nn.Softmax())

        self.fcn = nn.Sequential(*modules)

    def forward(self, x):
        return self.fcn(x)


def exp():
    fcn = FCN()
    input = torch.rand(784)

    print(fcn(input))


if __name__ == "__main__":
    exp()
