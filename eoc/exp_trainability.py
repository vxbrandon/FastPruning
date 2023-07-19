import torch
import torch.nn as nn
import numpy as np
import os
import wandb

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from meanfield import MeanField
from models.fcn import FCN
from custom_utils import utils


def d_tanh(x):
    """Derivative of tanh."""
    return 1.0 / np.cosh(x) ** 2


def init_weights(m, sw, sb):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=(np.sqrt(sw / m.out_features)))
        nn.init.normal_(m.bias, mean=0.0, std=np.sqrt(sb))


def exp_trainability(args=None) -> None:
    """
    Explore the trainability of FCN on MNISt after being initialized far from EOC curve.
    """

    # Parameters for experiments @Jay: Fix hard-coded
    data_type = "MNIST"
    depth = 30
    width = 200
    num_experiments = 2
    num_epochs = 30
    act = np.tanh
    d_act = d_tanh
    tau_1 = tau_2 = 1.0
    q_star = 0.5
    lr_rate = 1e-3
    optimizer = "ADAM"
    weight_decay = 5e-4

    # # Logs args
    # logs_args = {"log_dir": "log_dir", "params": f"width-{width}_depth-{depth}_q-{q_star}"}

    # Load dataset
    train_loader, test_loader, classes = utils.prepare_dataloader(
        data_type=data_type, is_flatten=True
    )
    num_classes = len(classes)
    input_dims = next(iter(train_loader))[0].size()[-1]

    # Pre-configuration
    seed = 1
    utils.set_random_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Model
    act_func = nn.Tanh
    fcn = FCN(
        num_layers=depth,
        hidden_dims=width,
        act_func=act_func,
        num_classes=num_classes,
        input_dims=input_dims,
    )
    for q_star in [0.2, 0.5, 1.0, 1.5]:
        print(f"Calculating eoc curve for qstar {q_star}...")
        meanfield = MeanField(np.tanh, d_act)
        sw, sb = meanfield.sw_sb(q_star, 1)
        group_name = f"depth-{depth}_width-{width}_q-{q_star}"  # For logging purpose
        config = {
            "depth": depth,
            "width": width,
            "epochs": num_epochs,
            "seed": seed,
            "activation": "tanh",
            "learning_rate": lr_rate,
            "optimizer": optimizer,
            "weight decay": weight_decay,
            "weight variance": sw,
            "bias variance": sb,
            "q_star": q_star,
            "tau": 0,
        }
        for tau in [-1.0, -0.5, -0.1,  0, 0.1,  0.5, 1.0]:
            wandb.init(
                project="Edge of Chaos",
                tags=["EOC preliminary", "EOC trainability"],
                group=group_name,
                name=f"tau={tau}",
            )
            config["tau"] = tau
            wandb.config = config
            new_sw = sw + tau
            fcn.apply(lambda m: init_weights(m, new_sw, sb))
            fcn, log_dict = utils.train_model(
                model=fcn,
                train_loader=train_loader,
                test_loader=test_loader,
                patience=10,
                num_epochs=num_epochs,
                verbose=True,
                device=device,
                is_log_dict=True,
                learning_rate=lr_rate,
                optimizer=optimizer,
            )
            train_accs = log_dict["train_acc"]
            eval_accs = log_dict["eval_acc"]
            epochs = log_dict["epochs"]

            train_accs_data = [[x, y] for (x, y) in zip(epochs, train_accs)]
            eval_accs_data = [[x, y] for (x, y) in zip(epochs, eval_accs)]
            train_table = wandb.Table(
                data=train_accs_data, columns=["epochs", "train accuracy"]
            )
            eval_table = wandb.Table(
                data=eval_accs_data, columns=["epochs", "eval accuracy"]
            )
            wandb.log(
                {
                    group_name
                    + "_train": wandb.plot.line(
                        train_table,
                        "epochs",
                        "train accuracy",
                        title="EOC curve trainability",
                    )
                }
            )
            wandb.log(
                {
                    group_name
                    + "_eval": wandb.plot.line(
                        eval_table,
                        "epochs",
                        "eval accuracy",
                        title="EOC curve trainability",
                    )
                }
            )
            wandb.run.summary["best_accuracy"] = eval_accs[-1]
            wandb.finish()


if __name__ == "__main__":
    exp_trainability()
