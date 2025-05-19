import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import piq

class CNN_AE(pl.LightningModule):
    def __init__(self,
                 anomaly_threshold=0.1,
                 loader_obj=None):
        super().__init__()

        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (16, 60, 60)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 30, 30)
            nn.ReLU(),
        )

        # Flatten
        self.flatten = nn.Flatten()

        # AE part
        self.encoder = None
        self._encoded_size = None
        self.decoder = None

        # Threshold for anomaly detection
        self.anomaly_threshold = anomaly_threshold
        self.loader_obj = loader_obj

        self.mse_metric = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self._cnn_out_shape = None

        self.save_hyperparameters()

    def forward(self, x):
        cnn_out = self.cnn(x)

        if self._cnn_out_shape is None:
            self._cnn_out_shape = cnn_out.shape[1:]  # (C, H, W)

            flattened_dim = cnn_out.numel() // cnn_out.shape[0]

            self.encoder = nn.Sequential(
                nn.Linear(flattened_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            ).to(x.device)

            self.decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, flattened_dim),
                nn.ReLU()
            ).to(x.device)

        flat = self.flatten(cnn_out)
        encoded = self.encoder(flat)
        decoded = self.decoder(encoded)
        return flat, decoded

    def training_step(self, batch, batch_idx):
        x, y = batch  # assume batch is (B, 1, w, w)
        x_flat, x_reconstructed = self.forward(x)
        loss = self.loss(x_reconstructed, x_flat, x, stage="train")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_flat, x_reconstructed = self.forward(x)
        loss = self.loss(x_reconstructed, x_flat, x, stage="test")
        return loss, x_flat, x_reconstructed

    def loss(self, x_reconstructed, x_flat, x, stage="train"):
        mse = self.mse_metric(x_reconstructed, x_flat)

        B = x.shape[0]
        C, H, W = self._cnn_out_shape

        x_flat_reshaped = x_flat.view(B, C, H, W)
        x_reconstructed_reshaped = x_reconstructed.view(B, C, H, W)

        ssim_value = self.ssim_metric(x_reconstructed_reshaped, x_flat_reshaped)
        psnr_value = piq.psnr(x_reconstructed_reshaped, x_flat_reshaped, data_range=1.0)

        loss = mse + (1 - ssim_value)

        self.log(f"{stage}_mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_psnr", psnr_value, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_ssim", ssim_value, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def calculate_roc_auc(self, batch):
        x, y = batch
        x_flat, x_reconstructed = self.forward(x)
        reconstruction_error = F.mse_loss(x_reconstructed, x_flat, reduction='none').view(x.size(0), -1).sum(dim=1)
        labels = (reconstruction_error > self.anomaly_threshold).float()  # Assuming anomaly is defined by reconstruction error

        auc = roc_auc_score(labels.cpu().numpy(), reconstruction_error.cpu().numpy())
        return auc

    def train_dataloader(self):
        return self.loader_obj.load_images_from_folder(type="train")

    def test_dataloader(self):
        return self.loader_obj.load_images_from_folder(type="test")
