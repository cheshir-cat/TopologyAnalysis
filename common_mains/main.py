from src.NetWorks.CNN_AE import CNN_AE
from src.loaders.DataLoad import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import webbrowser
import subprocess
import time


def run(window_sizes, threshold, metric, mode, max_epochs, log_dir):
    for window_size in window_sizes:
        print(f"\n=== Обработка окна {window_size} ===")

        # Пути и имена логов
        data_path = f"C:/Novikova/data/cutted/minus_5_param/{mode}_{metric}_{str(threshold).replace('.', '')}/Window_{window_size}"
        log_name = f"cutted5_{metric}_{mode}_{str(threshold).replace('.', '')}_{window_size}"

        # DataLoader и модель
        data_loader = DataLoader(data_path)
        model = CNN_AE(anomaly_threshold=0.05, loader_obj=data_loader)

        # Логгер
        logger = TensorBoardLogger(log_dir, name="", version=log_name)

        # === Определение max_epochs ===
        if window_size != 30:
            # Используем замер времени на предыдущих окнах
            est_epoch_time = prev_total_time / prev_epochs
            max_epochs = round(base_total_time / est_epoch_time)
            print(f" → Автоматически назначено эпох: {max_epochs}")

        # === Обучение ===
        start = time.time()

        trainer = pl.Trainer(max_epochs=max_epochs, logger=logger)
        trainer.fit(model, model.train_dataloader())
        trainer.test(model, model.test_dataloader())

        end = time.time()
        elapsed = end - start

        print(f"Время обучения: {elapsed:.2f} сек")

        # Сохраняем базовое время
        if window_size == 30:
            base_total_time = elapsed
            prev_total_time = elapsed
            prev_epochs = max_epochs
        else:
            prev_total_time = elapsed
            prev_epochs = max_epochs


if __name__ == "__main__":
    # Параметры
    window_sizes = [30]  #, 60, 90, 120]
    threshold = 0.5
    metric = "euclidean"
    mode = "binary"  # threshold binary
    max_epochs = 10
    log_dir = "lightning_logs"

    # Запуск
    run(window_sizes, threshold, metric, mode, max_epochs, log_dir)

    # Запуск TensorBoard
    print("\nЗапуск TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
    time.sleep(3)
    webbrowser.open("http://localhost:6006")
