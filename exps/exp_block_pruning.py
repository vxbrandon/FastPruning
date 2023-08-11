"""
Train Models
"""

import torch
import torch.nn as nn
import os
from typing import List

import custom_utils.utils as utils


def train(
    data_type: str = "MNIST",
    num_layers_list: List[int] = [5],
    seed: int = 0,
    lr_rate: float = 1e-3,
    num_epochs: int = 100,
    epoch_rewind: int =3,
    weight_decay: float = 5e-4,
    patience: int = 30,
):
    for num_layers in num_layers_list:
        utils.set_random_seeds(seed)
        cuda_device = torch.device("cuda:0")

        # Load dataset
        train_loader, test_loader, classes = utils.prepare_dataloader(
            num_workers=8,
            train_batch_size=128,
            test_batch_size=256,
            data_type=data_type,
            is_flatten=True,
        )
        input_dims = next(iter(train_loader))[0].size()[-1]

        # Initialize model
        model = utils.FCN(
            num_layers=num_layers, input_dims=input_dims, num_classes=len(classes)
        )
        model_filename = f"FCN_{num_layers}_{data_type}.pt"
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
        utils.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=cuda_device,
            optimizer="SGD",
            l2_regularization_strength=weight_decay,
            learning_rate=lr_rate,
            num_epochs=num_epochs,
            T_max=num_epochs,
            verbose=True,
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


# print("==SVHN==")
# train("SVHN")
# print("==FASHION MNIST")
# train("FASHION_MNIST")
# print("==CIFAR_10==")
# train("CIFAR_10")
# print("==MNIST==")
# train("MNIST")
