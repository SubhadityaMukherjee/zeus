#%%
# Imports

import argparse
import os
import sys

from albumentations.augmentations.crops.functional import crop
from albumentations.augmentations.geometric.resize import Resize

sys.path.append("../")
os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"

import glob
import tarfile

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn import metrics, model_selection, preprocessing
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import zeus
from zeus.callbacks import (EarlyStopping, GradientClipping, PlotLoss,
                            TensorBoardLogger)
from zeus.datasets import ImageDataset
from zeus.metrics import LabelSmoothingCrossEntropy, accuracy
from zeus.metrics.loss import psnr
from zeus.utils.model_helpers import *

#%%
# Defining

## Params

INPUT_PATH = "/media/hdd/Datasets/DenseHaze/"
MODEL_PATH = "./models/"
MODEL_NAME = os.path.basename("dense.pt")
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
IMAGE_SIZE = 64
UPSCALE_FACTOR = 2

#%%
class Model(zeus.Model):
    def __init__(self, num_classes, upscale_factor=2):
        super().__init__()
        self.num_classes = num_classes
        self.n_channels = 3

        self.upscale_factor = upscale_factor
        self.bilinear = True

        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, self.upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        # This is new
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)

        self._initialize_weights()
        self.network = nn.Sequential(
            self.convrelu(self.conv1),
            self.convrelu(self.conv2),
            self.convrelu(self.conv3),
            self.convrelu(self.conv4),
            self.pixel_shuffle,
        )

    def monitor_metrics(self, acc):
        return {"epoch": self.current_epoch}

    def convrelu(self, con):
        return nn.Sequential(con, nn.ReLU())

    def _initialize_weights(self):
        torch.nn.init.orthogonal_(
            self.conv1.weight, torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.orthogonal_(
            self.conv2.weight, torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.orthogonal_(
            self.conv3.weight, torch.nn.init.calculate_gain("relu")
        )
        torch.nn.init.orthogonal_(self.conv4.weight)

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        # batch_size, _, _ = image.shape

        outputs = self.network(image.float())

        if targets is not None:
            #  loss = nn.CrossEntropyLoss()(outputs, targets)
            # loss = LabelSmoothingCrossEntropy()(outputs, targets)
            # loss = nn.BCEWithLogitsLoss()(outputs, targets)
            loss = nn.MSELoss()(outputs, targets)
            # acc = accuracy(outputs, targets.long())
            # acc = psnr(outputs, targets)
            metrics = self.monitor_metrics(loss)
            # outputs = outputs.squeeze(0).cpu()
            outputs = torch.einsum("bchw-> bhw", outputs)
            self._callback_runner.callbacks[-1].writer.add_image(
                f"{str(self.current_epoch)}", outputs
            )
            # self.writer.add_image(outputs)
            return outputs, loss, metrics
        return outputs, 0, {}


#%%
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


crop_size = calculate_valid_crop_size(IMAGE_SIZE, UPSCALE_FACTOR)

train_aug = albumentations.Compose(
    [
        albumentations.CenterCrop(crop_size, crop_size),
        albumentations.Resize(crop_size // UPSCALE_FACTOR, crop_size // UPSCALE_FACTOR),
        ToTensorV2(),
    ],
    p=1.0,
)
train_aug_t = albumentations.Compose(
    [
        albumentations.CenterCrop(crop_size, crop_size),
        albumentations.Resize(crop_size, crop_size),
        ToTensorV2(),
    ],
    p=1.0,
)


#%%
# Training
class ImageClassDs(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform_i=None, transform_t=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_i = transform_i
        self.transform_t = transform_t
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("YCbCr"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("YCbCr"), dtype=np.float32)
        # image = np.array(Image.open(img_path).convert("YCbCr").split()[0])
        # mask = np.array(Image.open(mask_path).convert("YCbCr").split()[0])

        # mask[mask == 255.0] = 1.0

        # augmentations = self.transform(image=image, mask=mask)
        image = self.transform_i(image=image)["image"]
        mask = self.transform_t(image=mask)["image"]

        return {"image": image, "targets": mask}


#%%

dataset = ImageClassDs(
    image_dir=INPUT_PATH + "train",
    mask_dir=INPUT_PATH + "masks",
    transform_i=train_aug,
    transform_t=train_aug_t,
)

from torch.utils.data import random_split

n_val = int(len(dataset) * 0.3)
n_train = len(dataset) - n_val
train_dataset, valid_dataset = random_split(dataset, [n_train, n_val])

#%%
#  Callbacks

from torch.utils.tensorboard import SummaryWriter

model = Model(2, UPSCALE_FACTOR)

es = EarlyStopping(
    monitor="valid_loss",
    model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
    patience=3,
    mode="min",
)

tb = TensorBoardLogger("logs")
grc = GradientClipping(5)
pl = PlotLoss(30)

count_parameters(model, showtable=False)
#%%
EPOCHS = 30

model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_bs=TRAIN_BATCH_SIZE,
    valid_bs=VALID_BATCH_SIZE,
    device="cuda",
    epochs=EPOCHS,
    callbacks=[grc, pl, tb],
    fp16=False,
)
#%%
model.save(MODEL_PATH + "dense.pt")

# %%
def run_model(image, model):
    img = Image.open(image).convert("YCbCr")
    y, cb, cr = img.split()

    # model = torch.load(model)
    img_to_tensor = torchvision.transforms.ToTensor()
    # input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
    st = img_to_tensor(y)
    input = torch.stack([st, st, st], dim=1)
    input = input.cuda()

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode="L")

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge("YCbCr", [out_img_y, out_img_cb, out_img_cr]).convert("RGB")

    return out_img


run_model("/media/hdd/Datasets/BSDS300/images/test/69015.jpg", model.network)
