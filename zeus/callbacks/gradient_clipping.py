import numpy as np
import torch

from zeus import enums
from zeus.callbacks import Callback


class GradientClipping(Callback):
    def __init__(self, clip_value=5):
        self.clip_value = clip_value

    def on_optimizer_start(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
