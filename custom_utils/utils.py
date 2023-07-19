import os
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import sklearn.metrics
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

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
    eval_batch_size=256,
    data_type: str = "CIFAR_10",
    is_flatten: bool = False,
):
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                          std=(0.229, 0.224, 0.225))
    # ])

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                          std=(0.229, 0.224, 0.225))
    # ])

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

        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.SequentialSampler(test_set)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=eval_batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
        )

        classes = train_set.classes
    elif data_type == "MNIST":
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
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

        train_sampler = torch.utils.data.RandomSampler(train_set)
        test_sampler = torch.utils.data.SequentialSampler(test_set)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=eval_batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
        )
        classes = train_set.classes

    else:
        raise NotImplementedError(f"data_type {data_type} is not implemented.")

    return train_loader, test_loader, classes


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
    optimizer: str = "ADAM",
    l1_regularization_strength=0,
    l2_regularization_strength=1e-4,
    learning_rate=1e-3,
    num_epochs=200,
    patience: int = 5,
    T_max: int = 200,
    verbose: bool = False,
    is_log_dict: bool = False,
):
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    best_eval_loss = float("inf")

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=learning_rate,
    #                       momentum=0.9,
    #                       weight_decay=l2_regularization_strength)
    if optimizer == "ADAM":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=l2_regularization_strength,
            amsgrad=False,
        )
    elif optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_regularization_strength,
        )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=[100, 150],
    #                                                  gamma=0.1,
    #                                                  last_epoch=-1)
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

    if is_log_dict:
        log_dict = defaultdict(list)

    # Initialize best model
    best_model_state_dict = model.state_dict()
    patience_stack = 0

    for epoch in range(num_epochs):
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
        if eval_loss < best_eval_loss:
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
                "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
                    epoch + 1, train_loss, train_accuracy, eval_loss, eval_accuracy
                )
            )

        if is_log_dict:
            log_dict["epochs"].append(epoch)
            log_dict["train_loss"].append(train_loss)
            log_dict["eval_loss"].append(eval_loss)
            log_dict["train_acc"].append(train_accuracy)
            log_dict["eval_acc"].append(eval_accuracy)
    model.load_state_dict(best_model_state_dict)

    if is_log_dict:
        return model, log_dict
    else:
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
