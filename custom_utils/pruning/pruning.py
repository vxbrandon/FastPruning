import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import pandas as pd
from custom_utils.constants import STRUCTURED_PRUNING, UNSTRUCTURED_PRUNING, BLOCK_PRUNING
from custom_utils.utils import evaluate_model, train_model
from custom_utils.pruning.block_pruning import block_pruning

from typing import Dict


def pruning(model: nn.Module, pruning_type: str = UNSTRUCTURED_PRUNING, pruning_params: Dict = {}):
    # # If no parameters for pruning aren't given, use default pruning
    # if len(pruning_params) == 0:
    #     if pruning_type == UNSTRUCTURED_PRUNING:
    #         pruning_params = {"prune_amount": 0.2, ""}

    prune_amount = pruning_params["prune_amount"]
    if pruning_type == UNSTRUCTURED_PRUNING:
        params = pruning_params["params"]
        prune.global_unstructured(pruning_params=params, type=prune.L1Unstructured, amount=prune_amount)
    elif pruning_type == STRUCTURED_PRUNING:
        param_name = pruning_params["param_name"]
        dim = pruning_params["dim"]
        prune.ln_structured(module=model, name=param_name, amount= prune_amount, dim=dim)
    elif pruning_type == BLOCK_PRUNING:
        norm_order = pruning_params["norm_order"]
        block_size = pruning_params["block_size"]
        prune_module = pruning_params["prune_module"]  # Should be either Linear or Conv2d
        block_pruning(module=model, block_size=block_size, order=norm_order, module_type=prune_module)


def compute_final_pruning_rate(pruning_rate, num_iterations):
    """A function to compute the final pruning rate for iterative pruning.
        Note that this cannot be applied for global pruning rate if the pruning rate is heterogeneous among different layers.
    Args:
        pruning_rate (float): Pruning rate.
        num_iterations (int): Number of iterations.
    Returns:
        float: Final pruning rate.
    """

    final_pruning_rate = 1 - (1 - pruning_rate) ** num_iterations

    return final_pruning_rate


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    return num_zeros, num_elements, 0


def measure_global_sparsity(
        model, weight=True, bias=False, conv2d_use_mask=False, linear_use_mask=False
):
    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask
            )
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask
            )
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def iterative_pruning_finetuning(
        model,
        train_loader,
        test_loader,
        device,
        learning_rate,
        l1_regularization_strength,
        l2_regularization_strength,
        learning_rate_decay=0.1,
        conv2d_prune_amount=0.4,
        linear_prune_amount=0.2,
        num_iterations=10,
        num_epochs_per_iteration=10,
        model_filename_prefix="pruned_model",
        model_dir="saved_models",
        grouped_pruning=False,
        pruning_params: dict = {},
        is_stop_same_acc: bool = True,
):
    _, unpruned_eval_acc, _ = evaluate_model(
        model=model, test_loader=test_loader, device=device, criterion=None
    )
    TOLERANCE = 0.01  # Tolerance for early stopping pruning
    print(f"Unpruned evaluation acc is {unpruned_eval_acc}.")

    pruned_accuracies = []
    best_eval_accuracy = 0

    for i in range(num_iterations):
        print("=" * 20)
        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))

        if grouped_pruning == True:
            # Global pruning
            # I would rather call it grouped pruning.
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            prune_params = {}
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount,
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    if is_structured_pruning:
                        prune.ln_structured(
                            module,
                            name="weight",
                            amount=conv2d_prune_amount,
                            n=1,
                            dim=structured_dims,
                        )
                    else:
                        prune.l1_unstructured(
                            module, name="weight", amount=conv2d_prune_amount
                        )
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(
                        module, name="weight", amount=linear_prune_amount
                    )

        _, eval_accuracy, _ = evaluate_model(
            model=model, test_loader=test_loader, device=device, criterion=None
        )

        # classification_report = create_classification_report(
        #     model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model, weight=True, bias=False, conv2d_use_mask=True, linear_use_mask=False
        )

        print(f"Global Sparsity: {sparsity}")
        print("Conv2d Sparsity: ", 1 - (1 - conv2d_prune_amount) ** (i + 1))
        print("Test Accuracy: {:.3f}".format(eval_accuracy))

        # print(model.conv1._forward_pre_hooks)

        print("\nFine-tuning...")

        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            learning_rate=learning_rate * (learning_rate_decay ** i),
            num_epochs=num_epochs_per_iteration,
        )

        _, eval_accuracy, _ = evaluate_model(
            model=model, test_loader=test_loader, device=device, criterion=None
        )

        pruned_accuracies.append(eval_accuracy.cpu())

        # classification_report = create_classification_report(
        #     model=model, test_loader=test_loader, device=device)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model, weight=True, bias=False, conv2d_use_mask=True, linear_use_mask=False
        )

        print("Test Accuracy: {:.3f}".format(eval_accuracy))

        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy

        # model_filename = "{}_{}.pt".format(model_filename_prefix, i + 1)
        # model_filepath = os.path.join(model_dir, model_filename)
        # save_model(model=model,
        #            model_dir=model_dir,
        #            model_filename=model_filename)
        # model = load_model(model=model,
        #                    model_filepath=model_filepath,
        #                    device=device)
        if is_stop_same_acc:
            if unpruned_eval_acc - eval_accuracy > TOLERANCE:
                print("Stopping Pruning as it exceeds the tolerance.")
                return model, pruned_accuracies
        print("=" * 20)

    return model, pruned_accuracies


def remove_parameters(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model
