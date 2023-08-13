import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from custom_utils.pruning.diag_pruning import diag_pruning_linear, diag_pruning_conv2d
from custom_utils.pruning.block_pruning import block_pruning_linear


def linear_prune(
    module: nn.Linear, prune_type: str, block_size: int = 1, is_scaling: bool = False
):
    pruned_model_names = ["Block_diag", "Block_diag_perm", "Block_l1", "Unstructured"]
    prune_amount = float(max(0,  1 - block_size/min(module.weight.size())))
    k = block_size
    a = pruned_model_names

    if prune_type == a[0]:
        diag_pruning_linear(
            module=module,
            block_size=k,
        )
    elif prune_type == a[1]:
        diag_pruning_linear(module=module, block_size=k, perm_type="RANDOM")
    elif prune_type == a[2]:
        block_size = int(min(k, min(module.weight.size())))
        block_pruning_linear(
            module=module,
            block_size=(block_size, block_size),
            prune_amount=prune_amount,
        )
    elif prune_type == a[3]:
        prune.l1_unstructured(module, name="weight", amount=float(prune_amount))
    else:
        raise NotImplementedError
    if is_scaling and prune_type not in (a[2], a[3]):
        with torch.no_grad():
            for name, param in module.named_parameters():
                if "weight" in name:
                    param *= torch.sqrt(torch.tensor(1 / (1 - prune_amount), device=param.device))


def conv2d_prune(
    module: nn.Conv2d, block_size: int, prune_type: str, is_scaling: bool = False
):
    pruned_model_names = ["Block_diag", "Block_diag_perm", "Block_l1", "Unstructured"]
    a = pruned_model_names

    if prune_type == a[0]:
        diag_pruning_conv2d(
            module=module,
            block_size=block_size,
        )
    elif prune_type == a[1]:
        diag_pruning_conv2d(module=module, block_size=block_size, perm_type="RANDOM")
    elif prune_type == a[3]:
        min_channel = min(module.weight.size()[0], module.weight.size()[1])
        prune_amount = max(0, 1 - block_size / min_channel)
        prune.l1_unstructured(module, name="weight", amount=float(prune_amount))
    else:
        raise NotImplementedError(
            f"{prune_type} on Conv2d module is not implemented yet."
        )
