import os
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import sklearn.metrics
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from typing import List

from models.resnet import ResNet18


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(
    num_workers=8,
    train_batch_size=128,
    test_batch_size=256,
    data_type: str = "MNIST",
    is_flatten: bool = False,
    train_eval_prop: List[float] = [1.0, 0.0],
    seed: int = 0,
    is_resize_greyscale: bool = False,
):
    eval_batch_size = test_batch_size
    if data_type == "CIFAR_10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        if is_flatten:
            train_transform = transforms.Compose(
                [train_transform, transforms.Lambda(torch.flatten)]
            )
            test_transform = transforms.Compose(
                [test_transform, transforms.Lambda(torch.flatten)]
            )

        train_set = torchvision.datasets.CIFAR10(
            root="data", train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=test_transform
        )
        classes = train_set.classes
    elif data_type == "MNIST":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        )

        if is_resize_greyscale:
            train_transform = transforms.Compose(
                [
                    train_transform,
                    transforms.Resize((32, 32), antialias=True),
                    transforms.Lambda(lambda x: torch.cat([x] * 3, axis=0)),
                ],
            )
            test_transform = transforms.Compose(
                [
                    test_transform,
                    transforms.Resize((32, 32), antialias=True),
                    transforms.Lambda(lambda x: torch.cat([x] * 3, axis=0)),
                ],
            )

        if is_flatten:
            train_transform = transforms.Compose(
                [train_transform, transforms.Lambda(torch.flatten)]
            )
            test_transform = transforms.Compose(
                [test_transform, transforms.Lambda(torch.flatten)]
            )

        train_set = datasets.MNIST(
            root="data", train=True, download=True, transform=train_transform
        )
        test_set = datasets.MNIST(
            root="data", train=False, download=True, transform=test_transform
        )
        classes = train_set.classes
    elif data_type == "FASHION_MNIST":
        # Define the transformation
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # Normalize the tensor with mean and standard deviation
            ]
        )
        train_transform = test_transform = transform

        if is_resize_greyscale:
            train_transform = transforms.Compose(
                [
                    train_transform,
                    transforms.Resize((32, 32), antialias=True),
                    transforms.Lambda(lambda x: torch.cat([x] * 3, axis=0)),
                ],
            )
            test_transform = transforms.Compose(
                [
                    test_transform,
                    transforms.Resize((32, 32), antialias=True),
                    transforms.Lambda(lambda x: torch.cat([x] * 3, axis=0)),
                ],
            )
        if is_flatten:
            train_transform = transforms.Compose(
                [train_transform, transforms.Lambda(torch.flatten)]
            )
            test_transform = transforms.Compose(
                [test_transform, transforms.Lambda(torch.flatten)]
            )

        # Load the Fashion MNIST dataset
        train_set = torchvision.datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=train_transform,
        )

        test_set = torchvision.datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=test_transform,
        )
        classes = train_set.classes

    elif data_type == "SVHN":

        # Define the transformation
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_transform = test_transform = transform

        if is_flatten:
            train_transform = transforms.Compose(
                [train_transform, transforms.Lambda(torch.flatten)]
            )
            test_transform = transforms.Compose(
                [test_transform, transforms.Lambda(torch.flatten)]
            )

        # Load the SVHN dataset
        train_set = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=train_transform
        )

        test_set = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=test_transform
        )

        # Get the unique labels from the training set
        classes = sorted(list(set(train_set.labels.tolist())))

    else:
        raise NotImplementedError(f"data_type {data_type} is not implemented.")

    generator = torch.Generator().manual_seed(seed)
    train_set, eval_set = torch.utils.data.random_split(
        train_set, train_eval_prop, generator
    )
    train_sampler = torch.utils.data.RandomSampler(train_set)
    eval_sampler = torch.utils.data.SequentialSampler(eval_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_set,
        batch_size=train_batch_size,
        sampler=eval_sampler,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )

    if train_eval_prop[0] == 1:
        return train_loader, test_loader, classes
    else:
        return train_loader, eval_loader, test_loader, classes


def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    eval_time = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        start_time = time.time()
        outputs = model(inputs)
        eval_time += time.time() - start_time
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy, eval_time


def create_classification_report(model, device, test_loader):
    model.eval()
    model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in test_loader:
            y_true += data[1].numpy().tolist()
            images, _ = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.cpu().numpy().tolist()

    classification_report = sklearn.metrics.classification_report(
        y_true=y_true, y_pred=y_pred
    )

    return classification_report


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    l1_regularization_strength=0,
    l2_regularization_strength=1e-4,
    optimizer="ADAM",
    learning_rate=1e-2,
    num_epochs=200,
    patience: int = 5,
    T_max: int = 200,
    verbose: bool = False,
    epoch_rewind: int = 0,
    filepath_rewind: str = "",
):
    # The training configurations were not carefully selected.

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_eval_loss = float("inf")

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    if optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=l2_regularization_strength,
        )
    elif optimizer == "ADAM":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=l2_regularization_strength,
            amsgrad=False,
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer} is not implemented.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy, _ = evaluate_model(
        model=model, test_loader=test_loader, device=device, criterion=criterion
    )
    if verbose:
        print(
            "Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
                0, eval_loss, eval_accuracy
            )
        )

    # Initialize best model
    best_model_state_dict = model.state_dict()
    patience_stack = 0

    for epoch in range(num_epochs):
        # Save parameters for rewinding to inital parameters
        if epoch == epoch_rewind:
            if filepath_rewind != "":
                torch.save(model.state_dict(), filepath_rewind)

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            l1_reg = torch.tensor(0.0).to(device)
            for module in model.modules():
                mask = None
                weight = None
                for name, buffer in module.named_buffers():
                    if name == "weight_mask":
                        mask = buffer
                for name, param in module.named_parameters():
                    if name == "weight_orig":
                        weight = param
                # We usually only want to introduce sparsity to weights and prune weights.
                # Do the same for bias if necessary.
                if mask is not None and weight is not None:
                    l1_reg += torch.norm(mask * weight, 1)

            loss += l1_regularization_strength * l1_reg

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy, _ = evaluate_model(
            model=model, test_loader=test_loader, device=device, criterion=criterion
        )
        if eval_loss <= best_eval_loss:
            # best_model_state_dict = copy.deepcopy(model.state_dict())
            best_model_state_dict = model.state_dict()
            best_eval_loss = eval_loss
            patience_stack = 0
        else:
            patience_stack += 1

        if patience_stack >= patience:
            print("EARLY STOPPING")
            break

        # Set learning rate scheduler
        scheduler.step()

        if verbose:
            print(
                "Epoch: {:03d} Train Loss: {:.4f} Train Acc: {:.4f} Eval Loss: {:.4f} Eval Acc: {:.4f}".format(
                    epoch + 1, train_loss, train_accuracy, eval_loss, eval_accuracy
                )
            )
    model.load_state_dict(best_model_state_dict)
    return model


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model


def create_model(num_classes=10, model_func=ResNet18):  # torchvision.models.resnet18):
    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    # model = model_func(num_classes=num_classes, pretrained=False)
    model = model_func()

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def print_num_params(model):
    print(sum(p.numel() for p in model.parameters()))


def log_data(data: dict, filepath: str):
    log_pd = pd.DataFrame(data)
    log_pd.to_csv(filepath, index=False)


"""
Pruning Utils
"""


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
    is_structured_pruning: bool = False,
    structured_dims: int = 0,
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
            learning_rate=learning_rate * (learning_rate_decay**i),
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
