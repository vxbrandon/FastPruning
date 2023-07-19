import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import betterspy
from typing import Iterable, Callable


from custom_utils.constants import LINEAR, CONV2D

def diag_pruning_linear(
    module: nn.Linear,
    block_size: int = 4,
    perm_type: str = None,
):
    mask = torch.zeros_like(module.weight)
    assert (
        mask.size()[0] == mask.size()[1]
    ), "Diagonal pruning isn't implemented for rectangular matrix"

    num_rows = num_cols = mask.size()[0]
    num_blocks = min(num_rows // block_size, num_cols // block_size)
    residual_diagonal = min(num_rows % block_size, num_cols % block_size)

    for i in range(num_blocks):
        start_row = i * block_size
        end_row = start_row + block_size
        start_col = i * block_size
        end_col = start_col + block_size
        mask[start_row:end_row, start_col:end_col] = 1

    # If there is a residual diagonal, use a smaller block size to fit it
    if residual_diagonal > 0:
        start_row = num_blocks * block_size
        end_row = start_row + residual_diagonal
        start_col = num_blocks * block_size
        end_col = start_col + residual_diagonal
        mask[start_row:end_row, start_col:end_col] = 1

    if perm_type is not None:
        col_perm = torch.randperm(num_cols)
        row_perm = torch.randperm(num_rows)
        mask = mask[:, col_perm]
        mask = mask[row_perm, :]
    prune.custom_from_mask(module, "weight", mask)


def exp():
    layer1 = nn.Linear(100, 100)
    diag_pruning_linear(layer1, perm_type="random")
    print(layer1.weight)

    sparse_weight = sparse.csr_matrix(layer1.weight.detach())
    print(sparse_weight)

    import matplotlib
    matplotlib.use('TkAgg')
    plt.spy(sparse_weight)



if __name__ == "__main__":
    exp()
