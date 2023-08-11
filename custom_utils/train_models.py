"""
Train Models
"""

import torch
import torch.nn as nn
import os
from typing import List

import custom_utils.utils as utils
from models.fcn import FCN
from models.resnet import ResNet18
from models.vggnet import vgg19_bn


def train(
    data_type: str = "MNIST",
    model_type: str = "FCN",
    num_layers: int = 5,  # For FCN model only
    seed: int = 0,
    lr_rate: float = 1e-3,
    num_epochs: int = 100,
    epoch_rewind: int = 3,
    weight_decay: float = 5e-4,
    patience: int = 30,
    verbose: bool = True,
    model_dir: str = None,
):
    utils.set_random_seeds(seed)
    cuda_device = torch.device("cuda:0")

    # The CNN models assume that the dataset is of size (32, 32, 3), so we need to adapt the greyscale dataset to fit
    # this size.
    if (model_type == "FCN") or (data_type in ["CIFAR-10", "SVHN"]):
        is_resize_greyscale = False
    else:
        is_resize_greyscale = True

    # Flatten dataset if model_type is "FCN".
    is_flatten = True if model_type == "FCN" else False

    # Load dataset
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

    # Initialize model
    if model_type == "FCN":
        model = FCN(
            num_layers=num_layers, input_dims=input_dims, num_classes=len(classes)
        )
    elif model_type == "VGG-19":
        model = vgg19_bn()
    elif model_type == "RESNET-18":
        model = ResNet18()
    else:
        raise NotImplementedError(f"{model_type} is not implemented.")

    # Logging directory and filename
    model_filename = f"FCN_{num_layers}_{data_type}.pt"
    if model_dir is None:
        model_dir = "saved_models"
    model_filepath = os.path.join(model_dir, model_filename)

    if not (os.path.exists(model_dir)):
        os.makedirs(model_dir)

    if os.path.exists(model_filepath):
        print(
            "FCN is already trained. To create new pre-trained model, delete the existing model file."
        )
        return

    filepath_rewind = os.path.join(
        model_dir, f"FCN_{num_layers}_{data_type}_rewind_{epoch_rewind}.pt"
    )

    # Train model
    utils.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        device=cuda_device,
        optimizer="SGD",
        l2_regularization_strength=weight_decay,
        learning_rate=lr_rate,
        num_epochs=num_epochs,
        T_max=num_epochs,
        verbose=verbose,
        epoch_rewind=epoch_rewind,
        filepath_rewind=filepath_rewind,
        patience=patience,
    )

    utils.save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # utils.load_model(model, os.path.join(model_dir, model_filename), cuda_device)

    _, eval_accuracy, _ = utils.evaluate_model(
        model=model, test_loader=test_loader, device=cuda_device, criterion=None
    )
    print(f"Number of layers: {num_layers}/ Test Accuracy: {eval_accuracy}")


if __name__ == "__main__":
    dataset_types = ["MNIST", "CIFAR_10", "SVHN", "FASHION_MNIST"]
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
    print(model_dir)
    print("Pretraining Models...")

    # Pre-train FCN
    model_type = "FCN"
    for dataset_type in dataset_types:
        print(f"\n==Training {model_type} on {dataset_type}==")
        num_epochs = 300 if dataset_type == "CIFAR_10" else 100
        train(data_type=dataset_type, model_type=model_type, num_epochs=num_epochs)

    # Pre-train VGG-19bn
    model_type = "VGG-19"
    num_epochs = 300
    lr_rate = 5e-3
    for dataset_type in dataset_types:
        print(f"Training {model_type} on {dataset_type}..")
        num_epochs = 300 if dataset_type == "CIFAR_10" else 100
        train(
            data_type=dataset_type,
            model_type=model_type,
            num_epochs=num_epochs,
            lr_rate=lr_rate,
        )
