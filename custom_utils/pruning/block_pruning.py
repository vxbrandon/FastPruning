import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Iterable, Callable

from custom_utils.constants import LINEAR, CONV2D


def block_pruning_linear(
    module: nn.modules,
    block_size: torch.int64 = torch.tensor([4, 4]),
    prune_amount: float = 0.2,
    norm: Callable = np.linalg.norm,
    order: int = 1,
):
    assert len(block_size.size()) == 2, "Dimension of blocks of linear layer weights should be 2"

    # Calculate the average weight per block
    block_averages = []
    weight_matrix = module.weight.detach().numpy()
    if len(list(module.named_buffers())) == 0:
        prune_mask = np.ones_like(weight_matrix)
    else:
        prune_mask = list(module.named_buffers())[1]
    for row in range(0, weight_matrix.shape[0], block_size[0]):
        for col in range(0, weight_matrix.shape[1], block_size[1]):
            if prune_mask[row, col] == 0:  # Skip already pruned blocks
                continue
            block = weight_matrix[
                row: row + block_size[0], col : col + block_size[1]
            ]
            block_average = norm(block, order=order)
            block_averages.append(block_average)

    # Sort the block averages
    sorted_averages = np.sort(block_averages)

    # Determine the threshold for pruning
    threshold_index = int(len(sorted_averages) * prune_amount)
    threshold = sorted_averages[threshold_index]

    # Construct masks for pruning
    for row in range(0, weight_matrix.shape[0], block_size[0]):
        for col in range(0, weight_matrix.shape[1], block_size[1]):
            if prune_mask[row, col] == 0:  # Skip already pruned blocks
                continue
            block = weight_matrix[
                row : row + block_size[0], col : col + block_size[1]
            ]
            block_average = np.mean(block)
            if block_average <= threshold:
                mask_block = prune_mask[
                    row: row + block_size[0], col : col + block_size[1]
                ]
                mask_block.fill(0.0)
    prune.custom_from_mask(module, "weight", torch.tensor(prune_mask))


def block_pruning_conv2d(
    module: nn.modules,
    block_size: torch.int64 = torch.tensor([1, 3, 1]),
    prune_amount: float = 0.2,
    order: int = 1,
):
    # Weight is of form (out_channel, in_channels / groups, kernel_size[0], kernel_size[1])
    assert len(block_size.size()) == 3, "Dimension of blocks of conv2d weights should be 3"



def block_pruning(
    module: nn.modules,
    block_size: torch.int64,
    prune_amount: float = 0.2,
    order: int = 1,
    module_type: str = LINEAR,
):
    if type == LINEAR:
        block_pruning_linear(module, block_size, prune_amount, order=order)
    elif type == CONV2D:
        block_pruning_conv2d(module, block_size, prune_amount, order=order)
