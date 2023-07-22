import torch
import torch.nn as nn
import numpy as np
import os
import wandb
import random
import json
import betterspy
import scipy.sparse as sparse
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm

from typing import Tuple
from collections import defaultdict

from meanfield import MeanField
from models.fcn import FCN
from custom_utils import utils
from custom_utils.pruning.diag_pruning import diag_pruning_linear


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    """Derivative of tanh."""
    return 1.0 / np.cosh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    d = np.zeros_like(x)
    d[np.nonzero(x > 0)] = 1
    return d


def init_weights(m, sw, sb):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=(np.sqrt(sw / m.in_features)))
        nn.init.normal_(m.bias, mean=0.0, std=np.sqrt(sb))


def init_weights_new(m, sw, sb):
    if type(m) == nn.Linear:
        scaling_factor = m.in_features
        if hasattr(m, "weight_mask"):
            mask = m.weight_mask
            prune_amount = float(torch.sum(mask == 0) / torch.numel(mask))
            scaling_factor *= 1 - prune_amount
            nn.init.normal_(m.weight_orig, mean=0.0, std=(np.sqrt(sw / scaling_factor)))
        nn.init.normal_(m.bias, mean=0.0, std=np.sqrt(sb))


def get_act_func(act_func: str = "TANH") -> Tuple[nn.Module, callable, callable]:
    if act_func == "TANH":
        return nn.Tanh, tanh, d_tanh
    elif act_func == "RELU":
        return nn.ReLU, relu, d_tanh
    else:
        raise NotImplementedError(f"{act_func} is not implemented.")


def get_model(
    model: str = "FCN",
    input_dims: int = 784,
    num_classes: int = 10,
    depth: int = 5,
    width: int = 300,
    act_func: nn.Module = nn.Tanh,
    block_size: int = None,
) -> nn.Module:
    if model == "FCN":
        return FCN(
            input_dims=input_dims,
            hidden_dims=width,
            num_layers=depth,
            act_func=act_func,
            num_classes=num_classes,
        )
    elif model == "FCN_DIAG_PERM":
        fcn = FCN(
            input_dims=input_dims,
            hidden_dims=width,
            num_layers=depth,
            act_func=act_func,
            num_classes=num_classes,
        )
        assert (
            block_size is not None
        ), f"block_size should be defined to apply block diag pruning"
        for module in fcn.modules():
            if isinstance(module, nn.Linear) and module.out_features != num_classes:
                diag_pruning_linear(module, block_size=block_size, perm_type="RANDOM")
        return fcn
    else:
        raise NotImplementedError(f"model {model} is not implemented.")


def exp_trainability(args: argparse.Namespace = None) -> None:
    """
    Explore the trainability of FCN on MNISt after being initialized far from EOC curve.
    """
    # Load dataset
    train_loader, test_loader, classes = utils.prepare_dataloader(
        data_type=args.data_type, is_flatten=True
    )
    num_classes = len(classes)
    input_dims = next(iter(train_loader))[0].size()[-1]

    # Pre-configuration
    utils.set_random_seeds(args.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    nn_act_func, act_func, d_act_func = get_act_func(args.act_func)
    is_wandb = False
    is_plot = True

    # Define Model
    fcn = get_model(model=args.model,
                    input_dims=input_dims,
                    num_classes=num_classes,
                    depth=args.depth,
                    width=args.width,
                    act_func=nn_act_func,
                    block_size=args.block_size
                    )
    # Finding the list of tau percentages on which we do experiments
    min_tau_per, max_tau_per = args.tau_range
    tau_pers = np.logspace(min_tau_per, max_tau_per, args.num_taus).tolist()

    # Finding the list of q_stars on which we do experiments
    q_min = args.qstar_range[0]
    q_max = args.qstar_range[1]
    q_stars = np.linspace(q_min, q_max, args.num_qstars).tolist()

    # Lists for 3d graphs
    sw_grid = np.empty((len(tau_pers), len(q_stars)))
    sb_grid = np.empty((len(tau_pers), len(q_stars)))
    train_acc_grid = np.empty((len(tau_pers), len(q_stars)))
    eval_acc_grid = np.empty((len(tau_pers), len(q_stars)))

    for q_idx in range(len(q_stars)):
        q_star = q_stars[q_idx]

        # Calculating eoc curve
        print(f"Calculating eoc curve for qstar {q_star}...")
        meanfield = MeanField(act_func, d_act_func)
        sw, sb = meanfield.sw_sb(q_star, 1)
        group_name = f"{args.model}_depth-{args.depth}_width-{args.width}_q-{q_star}"  # For logging purpose

        # Logging hyperparameters to wandb config
        config = vars(args)
        config["weight_var"] = sw
        config["bias_var"] = sb

        for tau_idx in range(len(tau_pers)):
            tau_per = tau_pers[tau_idx]
            eval_acc = 0
            train_acc = 0
            new_sw = sw * tau_per
            sw_grid[tau_idx][q_idx] = new_sw
            sb_grid[tau_idx][q_idx] = sb
            for num_exp in range(args.num_exps):
                fcn.apply(lambda m: init_weights(m, new_sw, sb))
                fcn, log_dict = utils.train_model(
                    model=fcn,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    patience=args.patience,
                    num_epochs=args.epochs,
                    verbose=True,
                    device=device,
                    is_log_dict=True,
                    learning_rate=args.lr,
                    optimizer=args.optimizer,
                    l2_regularization_strength=args.weight_decay,
                )
                train_accs = log_dict["train_acc"]
                eval_accs = log_dict["eval_acc"]
                epochs = log_dict["epochs"]
                train_acc += train_accs[-1]
                eval_acc += eval_accs[-1]
            train_acc /= args.num_exps
            eval_acc /= args.num_exps
            train_acc_grid[tau_idx, q_idx] = train_acc
            eval_acc_grid[tau_idx, q_idx] = eval_acc

            if is_wandb:
                train_accs_data = [[x, y] for (x, y) in zip(epochs, train_accs)]
                eval_accs_data = [[x, y] for (x, y) in zip(epochs, eval_accs)]
                train_table = wandb.Table(
                    data=train_accs_data, columns=["epochs", "train accuracy"]
                )
                eval_table = wandb.Table(
                    data=eval_accs_data, columns=["epochs", "eval accuracy"]
                )
                config["tau_per"] = tau_per
                wandb.init(
                    project="Edge of Chaos",
                    tags=["EOC preliminary", "EOC trainability"],
                    group=group_name,
                    name=f"tau_per={tau_per}",
                    config=config,
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

            # Logging to csv files
            log_dir = os.path.join("logs", group_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            filename = f"tau_per-{tau_per}"
            filepath = os.path.join(log_dir, filename + ".csv")
            print(log_dict)
            utils.log_data(log_dict, filepath)

    if is_plot:
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = plt.axes(projection='3d')

        surf = ax.plot_surface(sw_grid, sb_grid, eval_acc_grid,)
                               # rstride=1, cstride=1, cmap=cm.coolwarm,
                               # linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel("sw")
        ax.set_ylabel("sb")
        ax.set_zlabel("eval acc")

        # EOC curve
        num_taus = sw_grid.shape[0]
        eoc_idx = (num_taus - 1) / 2
        eoc_sw_list = sw_grid[eoc_idx, :]
        eoc_sb_list = sb_grid[eoc_idx, :]
        eoc_eval_acc_list = eval_acc_grid[eoc_idx, :]
        ax.plot(eoc_sw_list, eoc_sb_list, eoc_eval_acc_list, label='EOC')
        plt.show()

        # logging in Wandb
        wandb.init(
            project="Edge of Chaos",
            tags=["EOC preliminary", "EOC trainability"],
            group="3d_graph",
            config=vars(args),
        )
        wandb.log({"3d plot": wandb.Image(fig)})
        wandb.finish()
    # Logging 3d results
    orig_log_dir = os.path.join("logs_3d", "run_")
    log_dir = orig_log_dir
    idx = 0
    while os.path.exists(log_dir):
        log_dir = orig_log_dir + str(idx)
        idx += 1
    os.makedirs(log_dir)
    params_dict = vars(args)
    graph_log_dict = {
        "sw_grid": sw_grid.tolist(),
        "sb_grid": sb_grid.tolist(),
        "train_acc_grid": train_acc_grid.tolist(),
        "eval_acc_grid": eval_acc_grid.tolist()
    }

    params_path = os.path.join(log_dir, "params.json")
    graph_log_path = os.path.join(log_dir, "3d_graph_log.json")
    with open(params_path, 'w+') as f:
        json.dump(params_dict, f)
    with open(graph_log_path, 'w+') as f:
        json.dump(graph_log_dict, f)

def plot_3d(filepath:str):
    with open(filepath, 'r') as f:
        loaded_dict = json.load(f)
    sw_grid = np.array(loaded_dict["sw_grid"])
    sb_grid = np.array(loaded_dict["sb_grid"])
    train_acc_grid = np.array(loaded_dict["train_acc_grid"])
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(sw_grid, sb_grid, train_acc_grid, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel("sw")
    ax.set_ylabel("sb")
    ax.set_zlabel("train acc")

    # EOC curve
    num_taus = sw_grid.shape[0]
    eoc_idx = int((num_taus - 1) / 2)
    eoc_sw_list = sw_grid[eoc_idx, :]
    eoc_sb_list = sb_grid[eoc_idx, :]
    eoc_eval_acc_list = train_acc_grid[eoc_idx, :]
    ax.plot(eoc_sw_list, eoc_sb_list, eoc_eval_acc_list, label='EOC')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # the following arguments are only for trainability experiment.
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--num_exps",
        default=2,
        type=int,
        help="Number of experiments to get average accuracy of trained model",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--model",
        default="FCN",
        type=str,
        help="Model type",
    )
    parser.add_argument(
        "--block_size",
        default=None,
        type=int,
        help="Block size for pruning models",
    )
    parser.add_argument(
        "--data_type",
        default="MNIST",
        type=str,
        choices=["MNIST", "CIFAR_10"],
        help="Model type",
    )
    parser.add_argument(
        "--act_func",
        default="TANH",
        type=str,
        help="Activation function for each layer in FCN",
    )
    parser.add_argument(
        "--num_taus",
        default=21,
        type=int,
        help="number of taus(multiplicative constant for variance of weight matrix to check the thickness of"
        "edge of chaos",
    )
    parser.add_argument(
        "--tau_range",
        default=(-1, 1),
        type=tuple,
        help="Range of taus(multiplicative constant) for variance of weight matrix (log10 scale)",
    )
    parser.add_argument(
        "--qstar_range",
        default=(0.1, 10.0),
        type=tuple,
        help="Range of taus(multiplicative constant) for variance of weight matrix",
    )
    parser.add_argument(
        "--num_qstars",
        default=10,
        type=int,
        help="Number of qstars with which we do experiments",
    )
    parser.add_argument(
        "--depth", default=20, type=int, help="Depth of FCN"
    )
    parser.add_argument(
        "--width", default=300, type=int, help="width of fully-connected layer"
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, help="batch size for SGD"
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="number of epochs to train FCN"
    )
    parser.add_argument("--optimizer", default="SGD", type=str, help="OPTIMIZER TYPE")
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="learning rate for training"
    )
    parser.add_argument(
        "--weight_decay", default=0, type=float, help="Weight decay for training"
    )
    parser.add_argument(
        "--patience",
        default=20,
        type=int,
        help="Number of epochs for which the model's train acc "
        "is allowed to be not decreasing.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="debug the main experiment"
    )
    args = parser.parse_args()
    assert args.num_taus % 2 != 0, f"{args.num_taus} should be odd to include tau_per = 1 which is EOC case"
    exp_trainability(args)
    # filepath = os.path.join("logs_3d", "run_", "3d_graph_log.json")
    # plot_3d(filepath)