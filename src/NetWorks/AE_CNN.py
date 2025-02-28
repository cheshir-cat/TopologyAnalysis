import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import matplotlib.pyplot as plt
import re
import numpy as np
import torch


from torch import optim
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support as score


from src.DataLoad import DataLoader


class AE(pl.LightningModule):
    def __init__(self):
        super(AE, self).__init__()

        self.loss_array = []

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, 5, 1, 0),
            nn.ReLU(),
            nn.Conv2d(3, 3, 5, 1, 0),
            nn.ReLU(),
            nn.Conv2d(3, 3, 5, 1, 0),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(3, 3, 5, 1, 4),
            nn.ReLU(),
            nn.Conv2d(3, 3, 5, 1, 4),
            nn.ReLU(),
            nn.Conv2d(3, 3, 5, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        #print("\n encoded: ", encoded.shape)
        decoded = self.decoder(encoded)
        #print("\n decoded: ", decoded.shape)

        return decoded


class AE_CNN_Model(pl.LightningModule):
    def __init__(self, path):
        super(AE_CNN_Model, self).__init__()

        self.path = path
        self.loader_obj = DataLoader(path)
        self.loss_array = []

        self.build_model()

    def build_model(self):
        self.conv = nn.Conv2d(3, 3, 5, stride=2, padding=2)

        # в зависимости от окна настраиваем слой
        if re.findall(r"\d+", self.path)[-1] == "30":
            self.fc1 = nn.Linear(12, 12)
        elif re.findall(r"\d+", self.path)[-1] == "60":
            self.fc1 = nn.Linear(48, 12)
        elif re.findall(r"\d+", self.path)[-1] == "90":
            self.fc1 = nn.Linear(75, 12)
        elif re.findall(r"\d+", self.path)[-1] == "120":
            self.fc1 = nn.Linear(147, 12)

        self.fc2 = nn.Linear(12, 1)

        self.auto_encoder = AE()

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        #print("x1: ", x.shape)
        x = self.pool(F.relu(self.conv(x)))
        #print("x2: ", x.shape)

        x = x.view(x.size(0), -1)
        #print("x view: ", x.shape)

        x = F.relu(self.fc1(x))
        #print("x fc1: ", x.shape)

        x = self.fc2(x)
        #print("x fc2: ", x.shape)

        x = x.squeeze()
        return x

    def training_step(self, x):
        x, y = x
        #print("\n x: ", x.shape, " y: ", y.shape)

        x = self.auto_encoder.forward(x)

        y_hat = self.forward(x)
        loss = self.loss(y.float(), y_hat)
        loss = loss.unsqueeze(0)

        output = OrderedDict({
            'loss': loss,
            'y': y,
            'y_hat': y_hat,
        })
        return output

    def predict(self, data):
        y_arr = []
        predicted_array = []

        for x, y in data:

            predicted = self.forward(x).round()

            y_arr.append(y)
            predicted_array.append(predicted)

        y_arr = torch.cat(y_arr)
        predicted_array = torch.cat(predicted_array)

        precision, recall, fscore, _ = score(y_arr.detach().numpy(), predicted_array.detach().numpy())
        print("precision: ", precision, "recall: ", recall, "fscore: ", fscore)

        return predicted_array

    def loss(self, y, restored):
        loss = F.mse_loss(y, restored)
        self.loss_array.append(float(loss))

        return loss

    def plot_loss(self):
        #self.auto_encoder.plot_loss()

        plt.plot(self.loss_array)
        plt.title("Loss graph")
        plt.xlabel("Epoch stage")
        plt.ylabel("Loss")
        plt.show()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return self.loader_obj.load_images_from_folder(type="train")

    def test_dataloader(self):
        return self.loader_obj.load_images_from_folder(type="test")
