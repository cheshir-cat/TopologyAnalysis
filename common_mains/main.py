from src.NetWorks.AE_CNN import AE_CNN_Model
from src.NetWorks.CNN_AE import CNN_AE_Model

import pytorch_lightning as pl
import torch

import sklearn
#from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold


trainer = pl.Trainer(
    accelerator="cpu" if not torch.cuda.is_available() else "gpu",
    max_epochs=3,
    enable_checkpointing=False
)

path = "C://Novikova/data/dataset_5/Window_90"

model = CNN_AE_Model(path=path)

print("TRAIN")
trainer.fit(model, model.train_dataloader())
model.plot_loss()

print("PREDICT")
model.predict(model.train_dataloader())
