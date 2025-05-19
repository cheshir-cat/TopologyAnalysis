import os
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import matplotlib.image
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame as df


def standartize(raw_data):
    scaler_standard = StandardScaler()
    standardized_data = scaler_standard.fit_transform(raw_data)
    df_standardized = pd.DataFrame(standardized_data, columns=raw_data.columns)
    return df_standardized


def normalize(raw_data):
    # Normalization
    scaler_minmax = MinMaxScaler()
    normalized_data = scaler_minmax.fit_transform(raw_data)
    df_normalized = pd.DataFrame(normalized_data, columns=raw_data.columns)

    return df_normalized


class RecurrencePlotGenerator:
    def __init__(
            self,
            load_dir: str = None,
            save_dir: str = None,
            threshold: Optional[float] = None,
            mode: Optional[str] = None,
            metric: Optional[str] = None,

            window: int = 60,
            p: float = 2.0,
            show_plot: bool = True,
            save_plot: bool = True,
            n_jobs: int = -1
    ):
        """
        Инициализация генератора рекуррентных графиков.

        Параметры:
            save_dir (str): Основная директория для сохранения графиков
            window: Размер окна
            threshold: Порог для бинарного RP
            mode: Режим ("binary" или "threshold")
            metric: Метрика расстояния
            p: Параметр для метрики Минковского
            show_plot: Показывать график
            save_plot: Сохранять график
            n_jobs: Количество ядер для параллельных вычислений (-1 = все ядра)
        """

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.window = window
        self.threshold = threshold
        self.mode = mode
        self.metric = metric
        self.p = p
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Создаем основную директорию если ее нет
        os.makedirs(self.save_dir, exist_ok=True)

    def change_save_dir(self, save_dir: str):
        self.save_dir = save_dir

    @staticmethod
    def _convert_to_numpy(data) -> np.ndarray:
        """Конвертирует входные данные в numpy array."""
        if isinstance(data, torch.Tensor):
            return data.numpy() if data.is_cuda else data.cpu().numpy()
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        if isinstance(data, list):
            return np.array(data)
        return data

    def _calculate_distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Вычисляет матрицу расстояний."""
        if self.metric == "minkowski":
            distances = pdist(x, metric=self.metric, p=self.p)
        else:
            distances = pdist(x, metric=self.metric)
        return squareform(distances)

    def _create_recurrence_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Создает матрицу рекуррентности."""
        if self.mode == "binary":
            return (distance_matrix <= self.threshold).astype(float)
        elif self.mode == "threshold":
            rp = np.clip(distance_matrix ** self.threshold, 0, 1)
            return rp
        else:
            raise ValueError(f"Неизвестный режим: {self.mode}")

    def _determine_save_subfolder(self, y: Optional[np.ndarray] = None) -> str:
        """Определяет подпапку для сохранения на основе процента единиц в y."""
        percent_ones = np.mean(y)
        return "Attack" if percent_ones > self.threshold else "Normal"

    def _plot_and_save(
            self,
            rp_matrix: np.ndarray,
            seq_num: int,
            y: Optional[np.ndarray] = None
    ) -> None:
        """Визуализирует и сохраняет рекуррентный график."""
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        ax.imshow(rp_matrix, cmap="binary" if self.mode == "binary" else "gray", origin="lower")

        if self.save_plot:
            subfolder = self._determine_save_subfolder(y)
            filename = f"RP_{seq_num}.png"
            save_path = os.path.join(self.save_dir, subfolder, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            matplotlib.image.imsave(save_path, rp_matrix, cmap="binary" if self.mode == "binary" else "gray")

        if self.show_plot:
            plt.show()
        plt.close()

    def _process_window(self, i: int, x_np: np.ndarray, y_np: np.ndarray) -> None:
        """Обрабатывает одно окно данных."""
        distance_matrix = self._calculate_distance_matrix(x_np[i - self.window:i])
        norm_matrix = (distance_matrix - np.min(distance_matrix)) / (
                np.max(distance_matrix) - np.min(distance_matrix))
        rp_matrix = self._create_recurrence_matrix(norm_matrix)
        self._plot_and_save(rp_matrix, i, y_np[i - self.window:i])

    def generate(
            self,
            x: Union[torch.Tensor, np.ndarray],
            y: Optional[Union[torch.Tensor, np.ndarray]]
    ) -> None:
        """
        Основной метод для генерации рекуррентных графиков с параллельной обработкой.

        Параметры:
            x: Входные данные [n_samples, n_features]
            y: Метки (0 или 1) [n_samples] (опционально)
        """
        x_np = self._convert_to_numpy(x)
        y_np = self._convert_to_numpy(y)

        # Создаем список индексов для обработки
        indices = range(self.window, x_np.shape[0], self.window)

        # Создаем частичную функцию для обработки одного окна
        process_func = partial(self._process_window, x_np=x_np, y_np=y_np)

        # Используем пул процессов для параллельной обработки
        with Pool(processes=self.n_jobs) as pool:
            for i, _ in enumerate(pool.imap(process_func, indices), 1):
                print(f'\rProgress: {i * self.window}/{x_np.shape[0]}, '
                      f'{round(i * self.window / x_np.shape[0] * 100, 3)}%', end='', flush=True)

    def load_and_prepare(self) -> None:
        """Загружает рекуррентный график и обрабатывает его."""

        raw_data_train = df(pd.read_csv(self.load_dir + "train_data.csv", sep=","))
        data_train = normalize(standartize(raw_data_train))

        X_train = data_train.iloc[:, :-1].values
        Y_train = data_train.iloc[:, -1:].values
        print("X_train: ", X_train.shape, Y_train.shape)

        raw_data_test = df(pd.read_csv(self.load_dir + "test_data.csv", sep=","))
        data_test = normalize(standartize(raw_data_test))

        X_test = data_test.iloc[:, :-1].values
        Y_test = data_test.iloc[:, -1:].values
        print("X_test: ", X_test.shape, Y_test.shape)

        self.generate(X_train, Y_train)

        self.change_save_dir(f"{self.load_dir}{self.mode}_{self.metric}_{str(self.threshold).replace('.', '')}/Window_{self.window}/test")
        self.generate(X_test, Y_test)
