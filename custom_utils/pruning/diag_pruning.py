from numpy.ma.extras import mask_cols
"""
Diagonal Block Pruning
"""
# !pip install betterspy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import numba as nb
import betterspy
from scipy import sparse
from typing import Iterable, Callable

def diag_pruning(weight: torch.tensor, mask:torch.tensor, block_size: int = 4):
    num_rows, num_cols = mask.shape

    # Compute the number of complete blocks and remaining diagonal size
    num_blocks = min(num_rows // block_size, num_cols // block_size)
    if num_blocks == 0:
        mask[:, :] = torch.ones_like(mask)
        return torch.ones_like(mask)

    if num_cols > num_rows:
        residual_diagonal = num_rows % block_size
    else:
        residual_diagonal = num_cols % block_size

    for i in range(num_blocks):
        block_dim = block_size if i < num_blocks - 1 else residual_diagonal

    # Create a mask to zero out non-block diagonal elements
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

    return mask

def diag_pruning_linear(
    module: nn.Linear,
    block_size: int = 4,
    perm_type: str = None,
    col_perm: torch.tensor = None,
    row_perm: torch.tensor = None,
):
    # mask = torch.zeros_like(module.weight)

    # num_rows, num_cols = mask.size()
    # num_blocks = min(num_rows // block_size, num_cols // block_size)
    # residual_diagonal = min(num_rows % block_size, num_cols % block_size)

    # for i in range(num_blocks):
    #     start_row = i * block_size
    #     end_row = start_row + block_size
    #     start_col = i * block_size
    #     end_col = start_col + block_size
    #     mask[start_row:end_row, start_col:end_col] = 1

    # # If there is a residual diagonal, use a smaller block size to fit it
    # if residual_diagonal > 0:
    #     start_row = num_blocks * block_size
    #     end_row = start_row + residual_diagonal
    #     start_col = num_blocks * block_size
    #     end_col = start_col + residual_diagonal
    #     mask[start_row:end_row, start_col:end_col] = 1
    assert isinstance(module, nn.Linear)
    mask = torch.zeros_like(module.weight)
    num_rows, num_cols = module.weight.shape
    max_size = max(num_rows, num_cols)
    min_size = min(num_rows, num_cols)

    num_reps = max_size // min_size + 1
    for j in range(num_reps):
      if num_rows > num_cols:
        diag_pruning(module.weight, mask[j * min_size:, :], block_size=block_size)
      else:
        diag_pruning(module.weight, mask[:, j * min_size:], block_size=block_size)
    if perm_type == "RANDOM":
        col_perm = torch.randperm(num_cols)
        row_perm = torch.randperm(num_rows)
        mask = mask[:, col_perm]
        mask = mask[row_perm, :]
    if perm_type == "CUSTOM":
        mask = mask[:, col_perm]
        mask = mask[row_perm, :]
    print(mask.shape, module.weight.shape)

    prune.custom_from_mask(module, "weight", mask)


def diag_pruning_conv2d(
    module: nn.Conv2d,
    block_size: int = 4,
    perm_type: str = None,
):
    assert isinstance(module, nn.Conv2d)
    in_out_channels = module.weight[:, :, 0, 0]

    mask = torch.zeros_like(in_out_channels)
    num_rows, num_cols = in_out_channels.shape
    max_size = max(num_rows, num_cols)
    min_size = min(num_rows, num_cols)

    num_reps = max_size // min_size + 1
    mask = torch.zeros_like(in_out_channels)
    for j in range(num_reps):
      if num_rows > num_cols:
        diag_pruning(in_out_channels, mask[j * min_size:, :], block_size=block_size)
      else:
        diag_pruning(in_out_channels, mask[:, j * min_size:], block_size=block_size)
    if perm_type == "RANDOM":
        col_perm = torch.randperm(num_cols)
        row_perm = torch.randperm(num_rows)
        mask = mask[:, col_perm]
        mask = mask[row_perm, :]
    conv2d_mask = torch.zeros_like(module.weight) # 4-dimensional
    non_zero_idcs = torch.nonzero(mask, as_tuple=True)
    conv2d_mask[non_zero_idcs] = 1
    # sparse_matrix = sparse.csr_matrix(mask.cpu().detach().numpy())
    # betterspy.show(sparse_matrix)


    prune.custom_from_mask(module, "weight", conv2d_mask)




def exp():
    layer1 = nn.Linear(37, 91)
    diag_pruning_linear(layer1, block_size=10, perm_type="RANDOM")


    # Show and save the sparsity pattern
    sparse_matrix = sparse.csr_matrix(layer1.weight.detach().numpy())
    betterspy.show(sparse_matrix)

    layer1 = nn.Linear(37, 91)
    diag_pruning_linear(layer1, block_size=10, perm_type="")

    # Show and save the sparsity pattern
    sparse_matrix = sparse.csr_matrix(layer1.weight.detach().numpy())
    betterspy.show(sparse_matrix)


if __name__ == "__main__":
    a = torch.rand((2,2,2))
    b = torch.rand((2,2))
    idcs = torch.nonzero(b<0.7, as_tuple=True)
    # a[idcs] = 0
    a[idcs] = 0
    # exp()
    a = torch.rand((10, 100, 100))
    c = nn.Conv2d(10, 10, kernel_size=3)
    d = c.weight.cpu().detach()
    b = c(a)
    diag_pruning_conv2d(c, block_size=15)
