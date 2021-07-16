import numpy as np
import torch
import torch.nn.utils.prune as prune

from zeus import enums
from zeus.callbacks import Callback

# class GradientClipping(Callback):
#     def __init__(self, clip_value=5):
#         self.clip_value = clip_value

#     def on_optimizer_start(self, model):
#         torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)

conv_names = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)


def pruner(
    model, grouped_pruning=True, conv2d_prune_amount=0.4, linear_prune_amount=0.2
):
    if grouped_pruning == True:
        # Global pruning
        parameters_to_prune = []
        for _, module in model.named_modules():
            if any([isinstance(module, x) for x in conv_names]):
                parameters_to_prune.append((module, "weight"))
        print(f"Pruning: {len(parameters_to_prune)}")
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=conv2d_prune_amount,
        )
    else:
        for _, module in model.named_modules():
            if any([isinstance(module, x) for x in conv_names]):
                prune.l1_unstructured(module, name="weight", amount=conv2d_prune_amount)
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=linear_prune_amount)


class PruningCallback(Callback):
    def __init__(
        self, grouped_pruning=True, conv2d_prune_amount=0.4, linear_prune_amount=0.2
    ):
        self.grouped_pruning = grouped_pruning
        self.conv2d_prune_amount = conv2d_prune_amount
        self.linear_prune_amount = linear_prune_amount

    def on_epoch_start(self, model):
        pruner(
            model,
            self.grouped_pruning,
            self.conv2d_prune_amount,
            self.linear_prune_amount,
        )
