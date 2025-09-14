# Phase 4 — Convolutional Neural Network (PyTorch)

![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

This repository contains **`Phase4-CNN-Hina-40457510.ipynb`**, a clean, end‑to‑end **CNN classifier** implemented in **PyTorch**/**TorchVision**. It follows an industry‑style supervised pipeline:

- **Data pipeline**: folder‑based datasets → `torchvision.datasets.ImageFolder` → `DataLoader`
- **Transforms & augmentation**: `torchvision.transforms` for train/val preprocessing
- **Model**: a custom **ConvNet** defined with `torch.nn` layers
- **Training loop**: forward → `CrossEntropyLoss` → backward → optimizer step
- **Evaluation**: overall accuracy and **confusion matrix**
- **Visualization**: training curves (loss/accuracy) + seaborn heatmaps

> The notebook is **dataset‑agnostic**. Put your images under `data/` (one subfolder per class) and update the dataset root if needed.

---

## Repository structure

```
.
├── Phase4-CNN-Hina-40457510.ipynb
├── data/
│   ├── train/
│   │   ├── class_a/ ... *.jpg|*.png
│   │   └── class_b/ ...
│   └── val/                 # optional; notebook can split from a single root
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

### 2) Prepare your dataset
Organize images in class‑named folders:
```
data/train/<class_name>/*.jpg
data/val/<class_name>/*.jpg       # or let the notebook create a split
```

### 3) Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```
Open **`Phase4-CNN-Hina-40457510.ipynb`** and run the cells top‑to‑bottom (Kernel → Restart & Run All).

---

## Training details

- **Device**: auto‑selects `cuda` (GPU), `mps` (Apple Silicon), or `cpu`
- **Loss**: `nn.CrossEntropyLoss` for multi‑class classification
- **Optimizers**: `torch.optim.Adam` or `SGD` (set in the notebook)
- **Batching**: `DataLoader` with shuffling for train, no shuffle for val
- **Metrics**: accuracy and confusion matrix (via scikit‑learn)

Example training loop pattern used:
```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    # compute validation metrics
```

---

## Reproducibility & tips

- Fix seeds:
  ```python
  import torch, random, numpy as np
  torch.manual_seed(42); random.seed(42); np.random.seed(42)
  ```
- Keep **the same transforms** for val/test across runs.
- Track runs by exporting metrics/plots to `outputs/`.
- For imbalanced datasets, add macro/weighted **F1** using scikit‑learn.

---

## Extending the notebook

- Add **learning‑rate schedulers** (`StepLR`, `ReduceLROnPlateau`)
- Introduce **early stopping** and **model checkpoints**
- Swap in **pretrained backbones** (`torchvision.models`) for fine‑tuning
- Log to **TensorBoard** or Weights & Biases for richer diagnostics

---

## Requirements

Install with:
```bash
pip install -r requirements.txt
```

**Core dependencies**
- `torch`, `torchvision`
- `scikit-learn` *(confusion matrix & accuracy)*
- `matplotlib`, `seaborn`
- `jupyter`
- `numpy`

> CUDA is optional. On macOS, MPS acceleration is used automatically if available.

---

## License
Choose a license (MIT/Apache‑2.0/BSD‑3‑Clause) and include a `LICENSE` file.

## Acknowledgements
- PyTorch & TorchVision
- scikit‑learn, numpy
- seaborn/matplotlib

---

**Maintainer tips**  
Clear outputs before committing to keep diffs tidy:
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "Phase4-CNN-Hina-40457510.ipynb"
```
Pin versions in `requirements.txt` for stable builds.
