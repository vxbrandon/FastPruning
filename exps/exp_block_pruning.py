"""
Train Models
"""

import torch
import torch.nn as nn
import os
import betterspy
import scipy.sparse as sparse
import numpy as np
from typing import List, Union, Iterable
from collections import defaultdict

import custom_utils.utils as utils
from custom_utils.pruning.pruning import linear_prune, conv2d_prune
from models.fcn import FCN
from models.resnet import ResNet18
from models.vggnet import vgg19_bn


def prune_and_finetune_linear(
    model_type: str = "FCN",
    num_layers: int = 5,
    data_type: str = "SVHN",
    model_dir: str = "saved_models",
    log_dir: str = "logs",
    optimizer: str = "ADAM",
    lr_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 10,
    num_epochs: int = 50,
    is_scaling: bool = True,
    is_iterative_finetune: bool = False,
    is_random_initialization: bool = False,
    verbose: bool = True,
    is_save_pruned_models: bool = False,
    is_log: bool = True,
    seed: int = 0,
    block_sizes: Iterable[int] = [1],
):
    # assert "FCN" in model_type, f"Experiments for pruning linear layers of {model_type} aren't implemented."
    # assert str(num_layers) in model_type, f"Given model type and number of layers don't match."

    # Set random seed
    utils.set_random_seeds(seed)

    # Device
    cuda_device = torch.device("cuda:0")

    # The CNN models assume that the dataset is of size (32, 32, 3), so we need to adapt the greyscale dataset to fit
    # this size.
    if (model_type == "FCN") or (data_type in ["CIFAR-10", "SVHN"]):
        is_resize_greyscale = False
    else:
        is_resize_greyscale = True

    # Flatten dataset if model_type is "FCN".
    is_flatten = True if model_type == "FCN" else False

    # Load datasets
    train_loader, val_loader, test_loader, classes = utils.prepare_dataloader(
        num_workers=8,
        train_batch_size=128,
        test_batch_size=256,
        data_type=data_type,
        is_flatten=is_flatten,
        is_resize_greyscale=is_resize_greyscale,
        train_eval_prop=[0.9, 0.1],
        seed=seed,
    )
    input_dims = next(iter(train_loader))[0].size()[-1]

    # Logs directory/ Filenames
    if model_type == "FCN":
        model_type += f"-{num_layers}"
    model_name = f"{model_type}_{data_type}"
    model_filename = model_name + ".pt"
    model_filepath = os.path.join(model_dir, model_filename)
    pruning_types = ["Block_diag", "Block_diag_perm", "Unstructured"]
    datas = [defaultdict(list) for _ in range(3)]
    pruned_model_filenames = []
    for idx in range(len(pruning_types)):
        filename = f"{pruning_types[idx]}_{model_name}"
        pruned_model_filenames.append(filename)

    # Prune using block diagonals
    pruned_models = [None] * len(pruning_types)

    # Pruning & Fine-tuning for given list of pruning amount
    for block_size in block_sizes:
        block_size = int(block_size)
        print(f"Block size: {block_size}")

        for idx in range(len(pruning_types)):
            # Initialize model
            if "FCN" in model_type:
                model = FCN(
                    num_layers=num_layers,
                    input_dims=input_dims,
                    hidden_dims=input_dims,
                    num_classes=len(classes),
                )
                prune_func = linear_prune
            elif model_type == "VGG-19":
                model = vgg19_bn()
                prune_func = conv2d_prune
            elif model_type == "RESNET-18":
                model = ResNet18()
                prune_func = conv2d_prune
            else:
                raise NotImplementedError(f"{model_type} is not implemented.")

            # Load pre-trained parameters if we don't conduct experiments of the effect of random initialization
            # after pruning
            if not is_random_initialization:
                model = utils.load_model(model, model_filepath, cuda_device)

            # Re-initialize pruned model
            pruning_type = pruning_types[idx]
            pruned_models[idx] = model

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.size()[0] != len(
                    classes
                ):
                    prune_func(
                        module=module,
                        prune_type=pruning_type,
                        block_size=block_size,
                        is_scaling=is_scaling,
                    )
                    if is_iterative_finetune:
                        utils.train_model(
                            model=model,
                            train_loader=train_loader,
                            test_loader=val_loader,
                            device=cuda_device,
                            learning_rate=lr_rate,
                            num_epochs=num_epochs,
                            T_max=num_epochs,
                            patience=patience,
                            verbose=verbose,
                            optimizer="ADAM",
                            l2_regularization_strength=weight_decay,
                        )

                    if verbose:
                        print(f"{name} is pruned.")
                        model.eval()
                        _, eval_accuracy, _ = utils.evaluate_model(
                            model=model,
                            test_loader=test_loader,
                            device=cuda_device,
                            criterion=None,
                        )
                        print(f"{pruning_types[idx]} eval accuracy: {eval_accuracy}\n")
            if not is_iterative_finetune:
                utils.train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=val_loader,
                    device=cuda_device,
                    learning_rate=lr_rate,
                    num_epochs=num_epochs,
                    T_max=num_epochs,
                    patience=patience,
                    verbose=verbose,
                    optimizer=optimizer,
                    l2_regularization_strength=weight_decay,
                )
            model.eval()
            _, eval_accuracy, _ = utils.evaluate_model(
                model=model, test_loader=test_loader, device=cuda_device, criterion=None
            )
            print(f"=={pruning_types[idx]} eval accuracy: {eval_accuracy}\n")
            _, _, sparsity = utils.measure_global_sparsity(model, linear_use_mask=True)
            print(f"==global sparsity: {sparsity}")

            data = datas[idx]
            data["block size"].append(float(block_size))
            data["global sparsity"].append(float(sparsity))
            data["eval_accuracy"].append(float(eval_accuracy.cpu()))

            if is_save_pruned_models:
                utils.remove_parameters(model)
                model_filename = pruned_model_filenames[idx] + f"_block_{block_size}.pt"
                utils.save_model(
                    model=model, model_dir=model_dir, model_filename=model_filename
                )

            for module in model.modules():
                if isinstance(module, nn.Linear) and module.weight.size()[0] != len(
                    classes
                ):
                    break

        # Log the results
        if is_log:
            for idx in range(len(pruning_types)):
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                filename = pruned_model_filenames[idx]
                if is_random_initialization:
                    filename += "_rand_init"

                if is_iterative_finetune:
                    filename += "_iterative"

                filepath = os.path.join(log_dir, filename + ".csv")
                print(datas[idx])
                utils.log_data(datas[idx], filepath)
    if verbose:
        print("\n==Sparse matrix plot")
        for pruned_model in pruned_models:
            for module in pruned_model.modules():
                if isinstance(module, nn.Linear) and module.weight.size()[0] != len(
                    classes
                ):
                    betterspy.show(sparse.csr_matrix(module.weight.cpu().detach()))
                    break


def get_block_sizes(model_type: str, data_type: str):
    if model_type in ["VGG-19", "RESNET-18"]:
        block_sizes = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    elif model_type == "FCN":
        if data_type in ["MNIST", "FASHION_MNIST"]:
            block_sizes = (
                np.geomspace(start=1, stop=784 / 2, num=9).astype(np.int64).tolist()
            )
        elif data_type in ["CIFAR_10", "SVHN"]:
            block_sizes = (
                np.geomspace(start=1, stop=3072 / 2, num=9).astype(np.int64).tolist()
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return block_sizes


def exp():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(parent_dir, "saved_models")
    log_dir = os.path.join(parent_dir, "logs", "pruning_fine-tuning_exps")
    model_types = ["FCN", "VGG-19"]
    data_types = ["MNIST", "CIFAR_10", "SVHN", "FASHION_MNIST"]
    num_layers = 5  # Hard-coded

    print("Pruning & Fine-tuning...")
    for model_type in model_types:
        if model_type == "FCN":
            weight_decay = 5e-4
        else:
            weight_decay = 1e-4
        for data_type in data_types:
            print(f"\n==model: {model_type} & dataset: {data_type}")
            block_sizes = get_block_sizes(model_type=model_type, data_type=data_type)
            prune_and_finetune_linear(
                model_type=model_type,
                data_type=data_type,
                model_dir=model_dir,
                log_dir=log_dir,
                num_layers=num_layers,
                weight_decay = weight_decay,
                block_sizes=block_sizes,
            )


if __name__ == "__main__":
    exp()
