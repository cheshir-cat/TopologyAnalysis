import pandas as pd
from pandas import DataFrame as df
import numpy as np

import matplotlib.image

import matplotlib.pyplot as plt
from collections import Counter

from pyts.multivariate.image import JointRecurrencePlot
from pyts.image import RecurrencePlot
from pyts.datasets import load_basic_motions

import torch
import torch.utils.data

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def getsamples_columns(data, sample_num, var_num, window, percentage=80):
    X = torch.zeros((sample_num, window, var_num - 1))
    Y = torch.zeros((sample_num, 1, 1))

    for i in range(sample_num):
        start = i
        end = i + window
        X[i, :, :] = data[start:end, :-1]
        tmp_Y = data[start:end, -1:]

        percent = sum(tmp_Y) / tmp_Y.shape[0]
        if percent > percentage / 100:
            Y[i, 0, 0] = 1
        else:
            Y[i, 0, 0] = 0

        return X, Y


def preprocessingRaw(X):
    # Standardization
    scaler_standard = StandardScaler()
    standardized_data = scaler_standard.fit_transform(X)
    df_standardized = pd.DataFrame(standardized_data, columns=X.columns)

    # Normalization
    scaler_minmax = MinMaxScaler()
    normalized_data = scaler_minmax.fit_transform(df_standardized)
    df_normalized = pd.DataFrame(normalized_data, columns=X.columns)

    return df_normalized


def drop_sensors(data):
    tmp_data = data.loc[data["Normal/Attack"] == 1]
    for col in data.columns:
        print("col name: ", col, " unique: ", tmp_data[col].unique().shape[0])

    data.drop([data.columns[3], data.columns[4],
               data.columns[9], data.columns[10], data.columns[11], data.columns[12], data.columns[13],
               data.columns[14], data.columns[15],
               data.columns[22], data.columns[23],
               data.columns[28], data.columns[29], data.columns[30], data.columns[31], data.columns[32],
               data.columns[41], data.columns[42],
               data.columns[47], data.columns[48], data.columns[49]], axis=1, inplace=True)

    print(data.columns)

    data.to_csv(path + "Novikova_no_sensors.csv", sep=",", index=False)


def unite_reccurrence_graphs(X, Y, path, idx, window):
    X = X.squeeze().T

    united = []
    for i in range(X.shape[0]):
        X_jrp = RecurrencePlot(threshold='point', percentage=50).fit_transform(X[i].squeeze().reshape(1, -1))
        X_jrp = torch.tensor(X_jrp).squeeze()
        united.append(X_jrp)

    tmp = united[0]
    for i in range(1, len(united)):
        tmp = tmp + united[i]

    united = tmp
    norms = np.linalg.norm(united, axis=1)
    united = (united / norms)
    #print("united: ", united.shape, "/n")

    """if Y[Y.shape[0] - 1, 0] == 1:
        matplotlib.image.imsave(path + f"Map_{window}/AttackMap/" + f"SWAT_{idx}.png", united, cmap='binary')
    else:
        matplotlib.image.imsave(path + f"Map_{window}/NormalMap/" + f"SWAT_{idx}.png", united, cmap='binary')"""


def RecurrenceGraph(X, Y, path, idx, window):
    # Recurrence plot transformation
    jrp = RecurrencePlot(threshold='point', percentage=50)
    X_jrp = jrp.fit_transform(X)
    X_jrp = torch.tensor(X_jrp).squeeze()

    if Y[0] == 1:
        matplotlib.image.imsave(path + f"Window_{window}/" + f"TrueMap/" + f"data_{idx}.png", X_jrp, cmap='binary')
    else:
        matplotlib.image.imsave(path + f"Window_{window}/" + f"FalseMap/" + f"data_{idx}.png", X_jrp, cmap='binary')
    return X_jrp


def RecurrenceGraph_single(X, path, idx, window):
    # Recurrence plot transformation
    jrp = RecurrencePlot(threshold='point', percentage=50)
    X_jrp = jrp.fit_transform(X)
    X_jrp = torch.tensor(X_jrp).squeeze()

    matplotlib.image.imsave(path + f"Window_{window}/" + f"data_{idx}.png", X_jrp, cmap='binary')
    return X_jrp


if __name__ == "__main__":
    path = "C:\LETI_work\Data_analys\data\RealData\origins\my_tmp/test/all/"

    window = 90

    data = df(pd.read_csv(path + "united_miner+chajnik+kompressor.csv", sep=","))
    print(data.columns)

    X = torch.tensor(data.iloc[:, 0]) #torch.tensor(data.iloc[:, 0:])
    #print(X.shape, Y.shape)

    #for i in range(window, X.shape[0], window):
    #    RecurrenceGraph(X[i - window:i].unsqueeze(0), Y[i - window:i], path, i, window)

    Y_final = data.iloc[:window:, 1:]
    for i in range(window, X.shape[0], window):
        RecurrenceGraph_single(X[i - window:i].unsqueeze(0), path, i, window)

    Y_final = df(Y_final, columns="class_chajnik class_kompressor class_miner".split())

    Y_final.to_csv(path + "Y.csv", index=False)

    #for i in range(window, X.shape[0], window):
    #    unite_reccurrence_graphs(X[i-window:i].unsqueeze(0), Y[i-window:i], path, i, window)
