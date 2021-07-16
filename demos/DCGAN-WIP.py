# Imports

#%%
import argparse
import os
import sys

sys.path.append("../")
os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"

import glob

import albumentations
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
from torch.nn import functional as F

import zeus
from zeus.callbacks import (EarlyStopping, GradientClipping, PlotLoss,
                            TensorBoardLogger)
from zeus.datasets import ImageDataset
from zeus.metrics import LabelSmoothingCrossEntropy
from zeus.utils.model_helpers import *

#%%
# Defining

# Params

INPUT_PATH = "/media/hdd/Datasets/animefacedataset/"
MODEL_PATH = "./models/"
MODEL_NAME = os.path.basename("blindness.pt")
TRAIN_BATCH_SIZE = 140
VALID_BATCH_SIZE = 140
IMAGE_SIZE = 64
#%% Models


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


#%%


class Model(zeus.Model):
    def __init__(self, nc, noise_dim, features_d, features_g):
        super().__init__()
        self.nc = nc
        self.noise_dim = noise_dim
        self.features_d = features_d
        self.features_g = features_g
        self.gen = Generator(self.noise_dim, self.nc, self.features_g)
        self.disc = Discriminator(self.nc, self.features_d)
        initialize_weights(self.gen)
        initialize_weights(self.disc)

    def monitor_metrics(self, lossd, lossg):
        return {"loss_D": lossd, "loss_G": lossg}

    def fetch_optimizer(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape
        criterion = nn.BCEWithLogitsLoss()

        image = image.to(self.device)
        noise = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
        fake = self.gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = self.disc(image).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = self.disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        self.disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        # opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = self.disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        self.gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        # opt_gen.step()
        metrics = self.monitor_metrics(loss_disc, loss_gen)
        return output, loss_gen, metrics

        # if targets is not None:
        #     #  loss = nn.CrossEntropyLoss()(outputs, targets)
        #     loss = LabelSmoothingCrossEntropy()(outputs, targets)
        #     metrics = self.monitor_metrics(outputs, targets)
        #     return outputs, loss, metrics
        # return outputs, 0, {}


# +
train_aug = albumentations.Compose(
    [
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Transpose(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)
#%% Data read

from pathlib import Path

from sklearn.model_selection import train_test_split

df = [x for x in Path.iterdir(Path(INPUT_PATH) / "anime")]

df = df[:1000]  # subset_test

train_images, valid_images = train_test_split(df, test_size=0.33)
train_labels, valid_labels = [1 for _ in range(len(train_images))], [
    1 for _ in range(len(valid_images))
]

#%%

#  Training
train_dataset = ImageDataset(
    image_paths=train_images,
    targets=train_labels,
    augmentations=train_aug,
)

valid_dataset = ImageDataset(
    image_paths=valid_images,
    targets=valid_labels,
    augmentations=valid_aug,
)

# Callbacks
nc = 3
noise_dim = 100
features_d = 64
features_g = 64
model = Model(nc, noise_dim=noise_dim, features_d=features_d, features_g=features_g)

es = EarlyStopping(
    monitor="valid_loss",
    model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
    patience=3,
    mode="min",
)

tb = TensorBoardLogger()
grc = GradientClipping(5)
pl = PlotLoss(2)
#%%
count_parameters(model, showtable=False)

#%%
EPOCHS = 2

model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_bs=TRAIN_BATCH_SIZE,
    valid_bs=VALID_BATCH_SIZE,
    device="cuda",
    epochs=EPOCHS,
    callbacks=[grc, pl, tb],
    #     callbacks=[es, tb],
    fp16=False,
)
# -

# %%

# %%
