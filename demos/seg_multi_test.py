import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.append("../")
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

import zeus

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"

INPUT_PATH = "/media/hdd/Datasets/CamVid/"
MODEL_PATH = "./models/"
MODEL_NAME = os.path.basename("seg.pt")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class Model(zeus.Model):
    def __init__(self, n_classes):
        super().__init__()

        self.model = UNET(3, n_classes).cuda()

    #     def monitor_metrics(self, outputs, targets):
    # #         accuracy = dice(targets, outputs)
    # #         return {"dice_score": float(accuracy)}
    #         return {"dice_score": 0.0}
    def monitor_metrics(self, outloss):
        return {"epoch": self.current_epoch, "ce_loss": float(outloss)}

    def fetch_optimizer(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape
        outputs = self.model(image)

        if targets is not None:
            targets = targets.long()
            targets = torch.argmax(targets, dim=3).long()
            #             print(outputs.shape, targets.shape)
            #             targets = targets.float().unsqueeze(1)
            #             loss = nn.BCEWithLogitsLoss()(outputs, targets)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(loss)
            return outputs, loss, metrics
        return outputs, 0, {}


def predicter(im_path, nc=32, model_path=Path(MODEL_PATH) / "new_mod_nclass"):
    mod = Model(nc)
    mod.load(model_path)
    mod.eval()
    mod = mod.model
    # load image
    tes = np.array(Image.open(im_path))
    tes = torch.tensor(tes).unsqueeze(0)
    tes = torch.einsum("bhwc->bchw", tes)
    #  print(tes.shape)

    # predict output
    test_output = mod(tes.float().cuda())

    #  test_output = torch.argmax(test_output, dim=1).long()
    #  print(test_output.shape)
    mod.train()

    #  inv_normalize = transforms.Normalize(
    #      mean= [-m/s for m, s in zip([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])],
    #      std= [1/s for s in [0.229, 0.224, 0.225]]
    #  )
    #  test_output = inv_normalize(test_output)
    #  test_output = torch.einsum('chw->hwc',test_output)
    return np.array(test_output.detach().cpu()).astype(np.uint8)
