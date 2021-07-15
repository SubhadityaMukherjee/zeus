# Imports
#%%
import argparse
import os
import sys

sys.path.append("../")
os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"

import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn import metrics, model_selection, preprocessing
from torch.nn import functional as F
from torch.utils.data import Dataset

import zeus
from zeus.callbacks import (EarlyStopping, GradientClipping, PlotLoss,
                            TensorBoardLogger)
from zeus.datasets import ImageDataset
from zeus.metrics import LabelSmoothingCrossEntropy, root_mean_squared_error
from zeus.utils.model_helpers import *

#%%
# Defining

## Params
#%%
INPUT_PATH = "/media/hdd/Datasets/faceKeypoint/"
MODEL_PATH = "./models/"
MODEL_NAME = os.path.basename("face.pt")
TRAIN_BATCH_SIZE = 140
VALID_BATCH_SIZE = 140
IMAGE_SIZE = 96
#%%
## Dataset
class FacialKeypointDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.category_names = [
            "left_eye_center_x",
            "left_eye_center_y",
            "right_eye_center_x",
            "right_eye_center_y",
            "left_eye_inner_corner_x",
            "left_eye_inner_corner_y",
            "left_eye_outer_corner_x",
            "left_eye_outer_corner_y",
            "right_eye_inner_corner_x",
            "right_eye_inner_corner_y",
            "right_eye_outer_corner_x",
            "right_eye_outer_corner_y",
            "left_eyebrow_inner_end_x",
            "left_eyebrow_inner_end_y",
            "left_eyebrow_outer_end_x",
            "left_eyebrow_outer_end_y",
            "right_eyebrow_inner_end_x",
            "right_eyebrow_inner_end_y",
            "right_eyebrow_outer_end_x",
            "right_eyebrow_outer_end_y",
            "nose_tip_x",
            "nose_tip_y",
            "mouth_left_corner_x",
            "mouth_left_corner_y",
            "mouth_right_corner_x",
            "mouth_right_corner_y",
            "mouth_center_top_lip_x",
            "mouth_center_top_lip_y",
            "mouth_center_bottom_lip_x",
            "mouth_center_bottom_lip_y",
        ]
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.train:
            # image = np.array(self.data.iloc[index, 30].split()).astype(np.float32)
            image = np.array(self.data.iloc[index, 30].split(), dtype=np.float32)
            labels = np.array(self.data.iloc[index, :30].tolist())
            labels[np.isnan(labels)] = -1
        else:
            image = np.array(self.data.iloc[index, 1].split()).astype(np.float32)
            labels = np.zeros(30)

        ignore_indices = labels == -1
        labels = labels.reshape(15, 2)

        if self.transform:
            image = np.repeat(image.reshape(96, 96, 1), 3, 2).astype(np.uint8)
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations["image"]
            labels = augmentations["keypoints"]

        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1

        return {"image": image, "targets": labels.astype(np.float32)}


#%%
## Model
from efficientnet_pytorch import EfficientNet


class Model(zeus.Model):
    def __init__(self):
        super().__init__()

        self.model = EfficientNet.from_pretrained("efficientnet-b0").cuda()
        self.model._fc = nn.Linear(1280, 30)

    def monitor_metrics(self, outputs, targets):
        accuracy = root_mean_squared_error(targets, outputs)
        return {"rmse": float(accuracy), "epoch": self.current_epoch}

    def fetch_optimizer(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape
        outputs = self.model(image)
        outputs[targets == -1] = -1

        if targets is not None:
            # targets = targets.float().unsqueeze(1)
            loss = nn.MSELoss(reduction="sum")(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, 0, {}


#%%

## Dataloading
#%%
train_aug = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.IAAAffine(shear=15, scale=1.0, mode="constant", p=0.2),
        A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
        A.OneOf(
            [
                A.GaussNoise(p=0.8),
                A.CLAHE(p=0.8),
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Posterize(p=0.8),
                A.Blur(p=0.8),
            ],
            p=1.0,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=0.8),
                A.CLAHE(p=0.8),
                A.ImageCompression(p=0.8),
                A.RandomGamma(p=0.8),
                A.Posterize(p=0.8),
                A.Blur(p=0.8),
            ],
            p=1.0,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            p=0.2,
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Normalize(
            mean=[0.4897, 0.4897, 0.4897],
            std=[0.2330, 0.2330, 0.2330],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)

valid_aug = A.Compose(
    [
        A.Resize(height=96, width=96),
        A.Normalize(
            mean=[0.4897, 0.4897, 0.4897],
            std=[0.2330, 0.2330, 0.2330],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
)

#%%
# check the trains
# from sklearn.model_selection import train_test_split

# df = pd.read_csv(Path(INPUT_PATH)/"training.csv")
# df["Image"] = df["Image"].astype(str)
# df_tr, df_ts = train_test_split(df, test_size=.3)
# df_tr.to_csv(Path(INPUT_PATH)/"train_split.csv")
# df_ts.to_csv(Path(INPUT_PATH)/"test_split.csv")

#%%

from torch.utils.data import DataLoader
from torchvision import transforms

train_ds = FacialKeypointDataset(Path(INPUT_PATH) / "training.csv", transform=train_aug)
valid_ds = FacialKeypointDataset(
    Path(INPUT_PATH) / "test.csv", train=False, transform=valid_aug
)

#%%
# Model paramas
model = Model()

es = EarlyStopping(
    monitor="valid_loss",
    model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
    patience=3,
    mode="min",
)

tb = TensorBoardLogger()
grc = GradientClipping(5)
pl = PlotLoss(10)

#%%
count_parameters(model, showtable=False)

#%%
EPOCHS = 100

model.fit(
    train_ds,
    valid_dataset=valid_ds,
    train_bs=TRAIN_BATCH_SIZE,
    valid_bs=TRAIN_BATCH_SIZE,
    device="cuda",
    epochs=EPOCHS,
    callbacks=[grc, pl, tb],
    fp16=False,
)

# %%
