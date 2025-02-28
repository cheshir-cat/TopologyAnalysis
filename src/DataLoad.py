import os
from torchvision import datasets, transforms
import torch
import numpy as np
import pandas as pd
from pandas import DataFrame as df

class DataLoader:
    def __init__(self, path):
        self.path = path

    def change_path(self, path):
        self.path = path

    def load_images_from_folder(self, type):

        if type == "train":
            dataset_folder = datasets.ImageFolder(self.path + "/train", transform=transforms.ToTensor())
        elif type == "test":
            dataset_folder = datasets.ImageFolder(self.path + "/test", transform=transforms.ToTensor())

        print("Len: ", len(dataset_folder), dataset_folder,
              set(dataset_folder.targets))

        dataloader = torch.utils.data.DataLoader(dataset_folder, batch_size=16, shuffle=True)
        return dataloader

    def load_train_images_multioutput(self, type_class, type_attack):

        tmp_path = self.path

        tmp_path += "/train"

        if type_class == "chajnik":
            tmp_path += "/chajnik"
        elif type_class == "miner":
            tmp_path += "/miner"
        elif type_class == "kompressor":
            tmp_path += "/kompressor"

        if type_attack == "mapped_single":
            tmp_path += "/mapped_single"
        elif type_attack == "mapped_double":
            tmp_path += "/mapped_double"

        dataset_folder = datasets.ImageFolder(tmp_path + "/Window_90", transform=transforms.ToTensor())

        print("Len: ", len(dataset_folder), dataset_folder,
              set(dataset_folder.targets))

        dataloader = torch.utils.data.DataLoader(dataset_folder, batch_size=16, shuffle=True)
        return dataloader

    def load_test_images_multioutput(self):

        tmp_path = self.path

        dataset_folder = datasets.ImageFolder(tmp_path + "/Window_90", transform=transforms.ToTensor())

        print("Len: ", len(dataset_folder), dataset_folder,
              set(dataset_folder.targets))

        import torch.utils.data as data_utils

        #train = data_utils.TensorDataset(features, targets)
        #data_y = df(pd.read_csv(tmp_path + "/Y.csv", sep=","))
        #data_y = torch.tensor(data_y.values)

        dataloader = torch.utils.data.DataLoader(dataset_folder, batch_size=16, shuffle=True)
        return dataloader