#%%
# # Imports

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
from PIL import Image
from sklearn import metrics, model_selection, preprocessing
from torch.autograd import Variable
from torch.nn import functional as F

import zeus
from zeus.callbacks import (EarlyStopping, GradientClipping, PlotLoss,
                            TensorBoardLogger)
from zeus.datasets import ImageDataset
from zeus.metrics import LabelSmoothingCrossEntropy
from zeus.utils.model_helpers import *

#%%
# # Defining

# ## Params

# INPUT_PATH = "/media/hdd/Datasets/blindness/"
MODEL_PATH = "./models/"
MODEL_NAME = os.path.basename("blindness.pt")
TRAIN_BATCH_SIZE = 140
VALID_BATCH_SIZE = 140
IMAGE_SIZE = 192

#%% CapsNet https://github.com/jindongwang/Pytorch-CapsuleNet/blob/master/capsnet.py
USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(
        self,
        num_capsules=8,
        in_channels=256,
        out_channels=32,
        kernel_size=9,
        num_routes=32 * 6 * 6,
    ):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=0,
                )
                for _ in range(num_capsules)
            ]
        )

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = (
            squared_norm
            * input_tensor
            / ((1.0 + squared_norm) * torch.sqrt(squared_norm))
        )
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(
        self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16
    ):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, out_channels, in_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(
                    u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1)
                )
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = (
            squared_norm
            * input_tensor
            / ((1.0 + squared_norm) * torch.sqrt(squared_norm))
        )
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_width * self.input_channel),
            nn.Sigmoid(),
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(
            dim=0, index=Variable(max_length_indices.squeeze(1).data)
        )
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(
            -1, self.input_channel, self.input_width, self.input_height
        )
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(
                config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size
            )
            self.primary_capsules = PrimaryCaps(
                config.pc_num_capsules,
                config.pc_in_channels,
                config.pc_out_channels,
                config.pc_kernel_size,
                config.pc_num_routes,
            )
            self.digit_capsules = DigitCaps(
                config.dc_num_capsules,
                config.dc_num_routes,
                config.dc_in_channels,
                config.dc_out_channels,
            )
            self.decoder = Decoder(
                config.input_width, config.input_height, config.cnn_in_channels
            )
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(
            data, reconstructions
        )

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(
            reconstructions.view(reconstructions.size(0), -1),
            data.view(reconstructions.size(0), -1),
        )
        return loss * 0.0005


#%%
class Model(zeus.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()

    def monitor_metrics(self, accuracy):
        return {"loss": accuracy, "epoch": self.current_epoch}

    def fetch_optimizer(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(
            reconstructions.view(reconstructions.size(0), -1),
            data.view(reconstructions.size(0), -1),
        )
        return loss * 0.0005

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape
        outputs = self.digit_capsules(self.primary_capsules(self.conv_layer(image)))
        reconstructions, masked = self.decoder(outputs, image)

        if targets is not None:
            loss = self.margin_loss(outputs, targets) + self.reconstruction_loss(
                image, reconstructions
            )

            metrics = self.monitor_metrics(loss)
            return outputs, loss, metrics
        return outputs, 0, {}


#%%

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
#%%

# ## Training
import torchvision


class Transforms:
    # Compatibility with image folder https://gitmemory.com/issue/albumentations-team/albumentations/879/824771225
    def __init__(self, transforms: albumentations.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


class cifar10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": img, "targets": target}


train_dataset = cifar10(
    "/media/hdd/Datasets/", transform=Transforms(train_aug), train=True
)
valid_dataset = cifar10(
    "/media/hdd/Datasets/", transform=Transforms(valid_aug), train=False
)
#%%
# ## Callbacks

model = Model(10)

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
# +
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
    fp16=True,
)
# -

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
