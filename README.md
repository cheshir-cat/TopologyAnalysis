# Recurrence Plots for Cyberattack Detection on SWaT Dataset

This project applies **Recurrence Plots (RP)** and a **CNN-based Autoencoder** to detect cyberattacks in industrial systems, specifically using the SWaT dataset.

## 📌 Overview

We transform time-series data into image-like recurrence plots that capture temporal dynamics. A convolutional autoencoder is then trained to reconstruct these plots, enabling anomaly detection based on reconstruction quality (MSE, SSIM, PSNR).

## 🔧 Technologies Used

- Python 3.10+
- PyTorch Lightning
- NumPy, Pandas, Matplotlib
- scikit-learn, SciPy
- TensorBoard
- Structural Similarity (torchmetrics)
- PSNR/SSIM (piq)
- SWaT dataset (secure water treatment system)

## 📁 Project Structure

├── data/ # Preprocessed RP images (Attack / Normal)
├── models/ # CNN-AE model files
├── scripts/
│ ├── RPlot.py # Recurrence Plot generator
│ ├── CNN_AE.py # CNN Autoencoder implementation
│ └── main.py # Training/testing routine
├── outputs/ # Training logs, metrics
└── Report_RP_Cyberattack.docx / .pdf

1. **Preprocessing:**
   - Normalization (MinMax / Z-score)
   - Removal of non-attacked features

2. **Recurrence Plot Generation:**
   - Time window sizes: 30, 60, 90, 120
   - Distance: Euclidean
   - Modes:
     - `binary`: thresholded at 0.5, 0.7, 0.9
     - `threshold`: powered transformation
   - Images saved into `Attack` / `Normal` folders.

3. **Model Architecture:**
   - Conv2D → ReLU → Conv2D → Flatten → Dense Encoder → Dense Decoder → Reshape
   - Loss: `MSE + (1 - SSIM)`
   - Metrics: MSE, SSIM, PSNR

4. **Dynamic Epoch System:**
   - Training duration is stabilized across varying window sizes using adaptive epoch calculation.

## 📊 Results

- **Best configuration:** `binary`, window=30, threshold=0.5
- SSIM: 0.54 | PSNR: 30.03 | MSE: 0.0010
- Model remained stable even when 2 non-attack features were removed
- Training time: ~1.3–2.3 mins per configuration
