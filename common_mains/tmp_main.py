import pandas as pd
from sklearn.model_selection import train_test_split
from pandas import DataFrame as df
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
from matplotlib import pyplot as plt

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


def prepare_simple_data():
    folder_path = "C://LETI_work/Data_analys/data/RealData/origins/my_tmp/test/orig_data"
    target_file = "майнер+чайник+компрессор2.csv"
    data = df(pd.read_csv(os.path.join(folder_path, target_file)))
    #data_kompressor.drop(data_kompressor.columns[1], axis=1, inplace=True)

    data.rename(columns={data.columns[0]: 'data'}, inplace=True)
    data = preprocessingRaw(data)

    """plt.plot(data.iloc[:500, 0])
    plt.title("Данные по компрессору")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()"""

    #target_file = "майнер1.csv"
    #data_miner = df(pd.read_csv(os.path.join(folder_path, target_file)))
    #data_miner.drop(data_miner.columns[1], axis=1, inplace=True)

    #data_miner.rename(columns={data_miner.columns[0]: 'data'}, inplace=True)
    #data_miner = preprocessingRaw(data_miner)

    #target_file = "чайник1.csv"
    #data_chajnik = df(pd.read_csv(os.path.join(folder_path, target_file)))
    #data_chajnik.drop(data_chajnik.columns[1], axis=1, inplace=True)

    #data_chajnik.rename(columns={data_chajnik.columns[0]: 'data'}, inplace=True)
    #data_chajnik = preprocessingRaw(data_chajnik)

    data["class_chajnik"] = 1
    data["class_kompressor"] = 1
    data["class_miner"] = 1

    #united = pd.concat([data_kompressor, data_miner, data_chajnik], axis=0)

    #print(united.shape)
    #united.fillna(0, inplace=True)
    #united.to_csv(os.path.join(folder_path, "mapped_single", "united_kompressor.csv"), index=False)
    data.to_csv(os.path.join("C:\LETI_work\Data_analys\data\RealData\origins\my_tmp/test/all",
                             "united_miner+chajnik+kompressor.csv"), index=False)


prepare_simple_data()

"""import matplotlib.image as img

testImage = img.imread('C:\LETI_work\Data_analys\data\RealData\origins\my_tmp/train\kompressor\mapped_single\Window_90\TrueMap/data_900.png')
# displaying the image
fig, ax = plt.subplots()
im = ax.imshow(testImage)
plt.title("Реккурентный график для компрессора")
plt.xlabel("Время")
plt.ylabel("Время")
plt.show()"""


"""data = df(pd.read_csv('C://Novikova/data/Novikova.csv'))

data.drop(data.columns[0], axis=1, inplace=True)

print(data.columns)

dataset_5 = data.copy()
dataset_5.drop([' AIT201',
                'FIT301',
                'AIT401',
                'AIT501',
                'FIT601'], axis=1, inplace=True)

dataset_6 = data.copy()
dataset_6.drop([' AIT201', 'AIT203',
                'FIT301', 'MV304',
                'AIT401', 'P402',
                'AIT501', 'FIT503',
                'FIT601', 'P603'], axis=1, inplace=True)

dataset_5.to_csv("C:/Novikova/data/dataset_5/dataset_5.csv", index=False)
dataset_6.to_csv("C:/Novikova/data/dataset_6/dataset_6.csv", index=False)


data5_train, data5_test = dataset_5.iloc[:int(dataset_5.shape[0]*0.8), :], dataset_5.iloc[int(dataset_5.shape[0]*0.8):, :]
print(data5_train.shape, data5_test.shape)

data5_train.to_csv('C:/Novikova/data/dataset_5/train_data.csv', index=False)
data5_test.to_csv('C:/Novikova/data/dataset_5/test_data.csv', index=False)

data6_train, data6_test = dataset_6.iloc[:int(dataset_6.shape[0]*0.8), :], dataset_6.iloc[int(dataset_6.shape[0]*0.8):, :]
print(data5_train.shape, data5_test.shape)

data6_train.to_csv('C:/Novikova/data/dataset_6/train_data.csv', index=False)
data6_test.to_csv('C:/Novikova/data/dataset_6/test_data.csv', index=False)"""
