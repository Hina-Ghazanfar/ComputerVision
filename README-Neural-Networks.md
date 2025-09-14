# Phase 3 — Neural Network Classifier (PyTorch)

![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

This repository contains **`Phase3-NN-Hina-40457510.ipynb`**, a clean, end‑to‑end **image/text classifier** built with **PyTorch**. It demonstrates a practical supervised learning workflow:

- **Data pipeline** with `torchvision.transforms` (augmentation) and `DataLoader`
- **Model** definition in `torch.nn` (CNN/MLP structure in the notebook)
- **Training loop** (forward → loss → backward → step)
- **Evaluation**: accuracy, **confusion matrix**, and **loss curves**
- **Visualization** with matplotlib/seaborn

> The notebook is dataset‑agnostic: point the data‑loading cell to your dataset under `data/` and adjust transforms / labels accordingly.

---

## Repository structure

```
.
├── Phase3-NN-Hina-40457510.ipynb
├── data/
│   ├── train/         # or place a CSV with paths+labels
│   └── val/           # optional; notebook can split from a single folder
├── requirements.txt
├── .gitignore
└── README.md
```

`data/` is Git‑ignored to keep private datasets out of the repo.

---

## Getting started

### 1) Clone & create a virtual environment
```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Add your dataset
Two common layouts are supported:
- **Folders per class** (images): `data/train/<class_name>/*.jpg` and `data/val/<class_name>/*.jpg`
- **CSV**: with columns like `filepath,label` (the notebook includes a small loader pattern)

Adjust the loading cell accordingly.

### 3) Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```
Open **`Phase3-NN-Hina-40457510.ipynb`** and run the cells top‑to‑bottom (Kernel → Restart & Run All).

---

## Training loop (pattern used)

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)  # e.g., CrossEntropyLoss
        loss.backward()
        optimizer.step()

    model.eval()
    # compute val loss/accuracy here
```

- **Loss:** `nn.CrossEntropyLoss` for multi‑class classification
- **Optimizers:** `torch.optim.Adam` / `SGD`
- **Augmentation:** `torchvision.transforms` (e.g., `RandomHorizontalFlip`, `ColorJitter`)

---

## Evaluation
The notebook reports:
- **Accuracy**
- **Confusion matrix** (seaborn heatmap)
- **Loss/accuracy curves** for training diagnostics

> For imbalanced data, add macro/weighted **F1** and **PR‑AUC** via scikit‑learn.

---

## Reproducibility & tips
- Set seeds for determinism (as far as possible in DL):
  ```python
  import torch, random, numpy as np
  torch.manual_seed(42); random.seed(42); np.random.seed(42)
  ```
- Use **Pipelines** for consistent preprocessing between train/val.
- Save artifacts (model weights, label encoder) under `outputs/`:
  ```python
  torch.save(model.state_dict(), "outputs/model.pt")
  ```
- On **Apple Silicon**, PyTorch supports **MPS** acceleration (device `"mps"`).

---

## Extending the notebook
- Add a **learning‑rate scheduler** (`StepLR`, `ReduceLROnPlateau`).
- Introduce **early stopping** and model checkpoints.
- Swap in **pretrained backbones** from `torchvision.models` and fine‑tune.
- Log metrics to **TensorBoard** or `wandb`.

---

## Requirements

Install with:
```bash
pip install -r requirements.txt
```

**Core dependencies**
- `torch`, `torchvision`
- `pandas` *(optional for CSV metadata)*
- `scikit-learn` *(confusion matrix, metrics)*
- `matplotlib`, `seaborn`
- `jupyter`
- `numpy`

> CUDA is optional. On macOS, MPS acceleration is used automatically if available.

---

## License
Choose a license (MIT/Apache‑2.0/BSD‑3‑Clause) and include a `LICENSE` file.

## Acknowledgements
- PyTorch & TorchVision
- scikit‑learn, pandas, numpy
- seaborn/matplotlib

---

**Maintainer tips**  
Clear outputs before committing to keep diffs tidy:
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "Phase3-NN-Hina-40457510.ipynb"
```
Pin versions in `requirements.txt` for stable builds.
