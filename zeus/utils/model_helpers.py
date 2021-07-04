import os
import random
from typing import *

import hiddenlayer as hl
import numpy as np
import torch
from prettytable import PrettyTable


def visualize_model(model, inp_size=[1, 3, 64, 64], device="cuda:0"):
    """
    Use hiddenlayer to visualize a model
    """
    return hl.build_graph(model, torch.zeros(inp_size).to(device))


def seed_everything(seed=42):
    """
    Seed everything with a number
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory():
    """
    Clear GPU cache
    """
    torch.cuda.empty_cache()


def listify(o):
    """
    Convert to list
    """
    if o is None:
        return []
    if isinstance(o, list):
        return o
    if isinstance(o, str):
        return [o]
    if isinstance(o, Iterable):
        return list(o)
    return [o]


def compose(x, funcs, *args, order_key="_order", **kwargs):
    """
    Chain functions
    """
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


def flatten(x):
    """
    Flatten tensor
    """
    return x.view(x.shape[0], -1)


def find_modules(m, cond):
    """
    Return modules with a condition
    """
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])


def is_lin_layer(l):
    """
    Check if linear
    """
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.ReLU)
    return isinstance(l, lin_layers)


def count_parameters(model, showtable=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if showtable == True:
        print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
