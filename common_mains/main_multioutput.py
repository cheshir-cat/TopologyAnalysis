from src.NetWorks.CNN_AE_Multioutput import CNN_AE_Multioutput_Model

import pytorch_lightning as pl
import torch

import numpy as np
from pandas import DataFrame as df
import pandas as pd

#import sklearn
#from sklearn.ensemble import BaggingClassifier
#from sklearn.model_selection import KFold
#from sklearn.multioutput import MultiOutputRegressor
#from sklearn.ensemble import VotingClassifier

from sklearn.metrics import precision_recall_fscore_support as score


def form_basic_binar_classifiers(path):
    model_chajnik = CNN_AE_Multioutput_Model(path=path, type_class="chajnik", type_attack="mapped_single")
    model_miner = CNN_AE_Multioutput_Model(path=path, type_class="miner", type_attack="mapped_single")
    model_kompressor = CNN_AE_Multioutput_Model(path=path, type_class="kompressor", type_attack="mapped_single")

    models = [model_chajnik, model_kompressor, model_miner]

    return models


def train_models_single(models):
    trained_models = []
    for model in models:
        print("TRAINING SINGLE")
        trainer = pl.Trainer(
            accelerator="cpu" if not torch.cuda.is_available() else "gpu",
            max_epochs=3,
            enable_checkpointing=False
        )

        trainer.fit(model, model.train_dataloader())
        model.plot_loss()

        trained_models.append(model)

    return trained_models

def train_models_double(models):
    trained_models = []
    for model in models:
        print("TRAINING DOUBLE")
        trainer = pl.Trainer(
            accelerator="cpu" if not torch.cuda.is_available() else "gpu",
            max_epochs=3,
            enable_checkpointing=False
        )

        model.change_type_attack("mapped_double")
        trainer.fit(model, model.train_dataloader())
        model.plot_loss()

        trained_models.append(model)

    return trained_models

def test_models(models, test_path):
    predicted_array = []

    for model in models:
        print("TESTING ")
        model.change_path(test_path)
        preds = model.predict(model.test_dataloader())
        predicted_array.append(preds)

    #print("\n predicted_array final: ", len(predicted_array), len(predicted_array[0]), predicted_array)
    predicted_array = torch.tensor(torch.stack(predicted_array)).T

    y_arr = df(pd.read_csv(test_path + "/Y.csv", sep=","))
    y_arr = torch.tensor(y_arr.values)

    if predicted_array.shape[0] != y_arr.shape[0]:
        predicted_array = predicted_array[:y_arr.shape[0], :]
        y_arr = y_arr[:predicted_array.shape[0], :]
    print("\n predicted_array final: ", predicted_array.shape, predicted_array[0])
    print("\n y_arr: ", y_arr.shape)

    return y_arr, predicted_array


path = "C://LETI_work/Data_analys/data/RealData/origins/my_tmp"

models = form_basic_binar_classifiers(path)

models = train_models_single(models)
models = train_models_double(models)

path_test = "C://LETI_work/Data_analys/data/RealData/origins/my_tmp/test/all"
y_arr, predicted_array = test_models(models, path_test)

precision, recall, fscore, _ = score(predicted_array.detach().numpy(), y_arr)
print("precision: ", precision, "recall: ", recall, "fscore: ", fscore)


