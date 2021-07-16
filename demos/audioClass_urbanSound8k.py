#%%
# Imports

import argparse
import os
import sys

sys.path.append("../")
os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"

import glob
import tarfile

import albumentations
#%% Audio imports
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import zeus
from zeus.callbacks import (EarlyStopping, GradientClipping, PlotLoss,
                            TensorBoardLogger)
from zeus.datasets import ImageDataset
from zeus.metrics import LabelSmoothingCrossEntropy, accuracy
from zeus.utils.model_helpers import *

#%%
# Defining

## Params

INPUT_PATH = "/media/hdd/Datasets/UrbanSound8K"
MODEL_PATH = "./models/"
MODEL_NAME = os.path.basename("urban.pt")
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128
IMAGE_SIZE = 192

#%%
class Model(zeus.Model):
    def __init__(self, num_classes, input_size=40):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Tanh(),
        )

    def monitor_metrics(self, acc):
        # return {"accuracy": acc,"epoch": self.current_epoch}
        return {"epoch": self.current_epoch}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        # batch_size, _, _ = image.shape

        outputs = self.network(image)

        if targets is not None:
            #  loss = nn.CrossEntropyLoss()(outputs, targets)
            # loss = LabelSmoothingCrossEntropy()(outputs, targets)
            outputs = torch.einsum("bxy->byx", outputs)
            loss = F.cross_entropy(outputs, targets.long())
            acc = accuracy(outputs, targets.long())
            metrics = self.monitor_metrics(acc)
            return outputs, loss, metrics
        return outputs, 0, {}


#%%
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
def extract_mfcc(path):
    audio, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(audio, sr, n_mfcc=40)
    return torch.tensor(np.mean(mfccs.T, axis=0))


#  Data pre process

df = pd.read_csv(INPUT_PATH + "/metadata/UrbanSound8K.csv")
df.head(3)
print(df.shape)

# SUBSET REMOVE LATER
df = df.head(2000)

#%% get audio data and save to csv

# run this bit only once
from multiprocessing import Pool

num_classes = len(list(df["class"].unique()))


def parallelize_dataframe(df, func, n_cores=8):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def apply_mfcc(df):
    df["audio"] = df["full_file_name"].apply(extract_mfcc)
    return df


# df["full_file_name"] = INPUT_PATH + "/audio/"+df["slice_file_name"]

# df = parallelize_dataframe(df, apply_mfcc)
# df.to_pickle(INPUT_PATH + "/metadata/processed.pkl")

#%% read from saved data
df = pd.read_pickle(INPUT_PATH + "/metadata/processed.pkl")

#%%

from sklearn.model_selection import train_test_split

train_images, valid_images = train_test_split(
    df, test_size=0.33, stratify=df["classID"]
)

train_image_paths, valid_image_paths = (
    train_images["audio"].values,
    valid_images["audio"].values,
)

train_targets, valid_targets = (
    train_images["classID"].values,
    valid_images["classID"].values,
)

#%%
# Training


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"image": self.x, "targets": self.y}


#%%
train_dataset = AudioDataset(
    torch.stack([x for x in train_image_paths]),
    torch.stack([torch.tensor(x) for x in train_targets]),
)
valid_dataset = AudioDataset(
    torch.stack([x for x in valid_image_paths]),
    torch.stack([torch.tensor(x) for x in valid_targets]),
)

print(len(train_dataset), len(valid_dataset))
#%%
#  Callbacks
model = Model(10)

es = EarlyStopping(
    monitor="valid_loss",
    model_path=os.path.join(MODEL_PATH, MODEL_NAME + ".bin"),
    patience=3,
    mode="min",
)

tb = TensorBoardLogger()
grc = GradientClipping(5)
pl = PlotLoss(30)

count_parameters(model, showtable=False)
#%%
EPOCHS = 50

model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_bs=TRAIN_BATCH_SIZE,
    valid_bs=VALID_BATCH_SIZE,
    device="cuda",
    epochs=EPOCHS,
    callbacks=[grc, pl, tb],
    fp16=True,
)
# -

# %%
# %%
