# Hand‑Crafted Features for Image Classification — Phase‑1/Phase‑2 Pipeline

![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blueviolet)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

This repository provides **`Phase1-Phase2-HandCraftedFeatures.ipynb`**, a practical **Bag of Visual Words (BoVW)** workflow for image classification using **hand‑crafted features**. It implements:

- Local feature extraction with **SIFT** and/or **ORB** (OpenCV)
- **Visual vocabulary** learning via **KMeans**
- Image representation as **histograms of visual words**
- A linear **SVM** classifier (`LinearSVC`) trained on BoVW features
- **Evaluation** with confusion matrix and classification report (seaborn/matplotlib)

The notebook is organized as a **two‑phase** pipeline:

- **Phase 1 — Feature Extraction & Vocabulary Build**  
  Detect local descriptors (SIFT/ORB), aggregate, and fit **KMeans** to create a codebook.
- **Phase 2 — Encoding & Classification**  
  Quantize each image’s descriptors to nearest codewords → build **BoVW histograms** → train **LinearSVC** → evaluate.

---

## Repository structure

```
.
├── Phase1-Phase2-HandCraftedFeatures.ipynb
├── data/
│   ├── train/
│   │   ├── class_a/ ... images ...
│   │   └── class_b/ ... images ...
│   └── test/             # optional; else notebook makes a split
├── requirements.txt
├── .gitignore
└── README.md
```

> **Tip:** Keep your dataset in `data/` with **one folder per class**. The notebook can also create a train/val split if you only provide a single dataset root.

---

## Getting started

### 1) Clone and create a virtual environment
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

### 2) Prepare images
Place images under `data/` as shown above. Supported formats include `.jpg`, `.png`, etc.

### 3) Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```
Open **`Phase1-Phase2-HandCraftedFeatures.ipynb`** and run cells top‑to‑bottom (Kernel → Restart & Run All).

---

## Pipeline details

### Feature extraction (OpenCV)
- **SIFT**: `cv2.SIFT_create()` descriptors (robust, scale/rotation‑invariant)
- **ORB**: `cv2.ORB_create()` descriptors (fast, binary)
- You can toggle which detector/describer to use in the relevant cell.

### Visual vocabulary (KMeans)
- Stack descriptors across training images
- Fit **`KMeans(n_clusters=K, random_state=42)`**
- `K` controls codebook granularity; start with 100–400 for small datasets

### Encoding (BoVW histogram)
- Assign each descriptor to nearest cluster center
- Build a **K‑dimensional histogram** per image (optionally L2‑normalize)
- Concatenate into feature matrix `X`

### Classification
- Train **`LinearSVC`** on `(X, y)`
- Evaluate with **`classification_report`** and **confusion matrix** (seaborn heatmap)

---

## Reproducibility & tips

- Set seeds for KMeans and any splits (`random_state=42`).
- Keep feature scaling consistent (binary ORB usually doesn’t need scaling; SIFT histograms often benefit from L2‑norm).
- **Class imbalance:** report macro/weighted F1 if classes are imbalanced.
- **Speed:** Use **MiniBatchKMeans** for large descriptor sets.
- **Persistence:** Save the fitted KMeans and SVM to reuse on new images.

---

## Extending the notebook

- Add **HOG/LBP/GLCM** descriptors for texture‑heavy datasets.
- Replace `LinearSVC` with **SVC (RBF)** or **RandomForestClassifier** for non‑linear boundaries.
- Add **grid search** with `GridSearchCV` to tune `K`, SVM `C`, and normalization.
- Export predictions and metrics to `outputs/` for experiment tracking.

---

## Requirements

Install with:
```bash
pip install -r requirements.txt
```

**Core dependencies**
- `opencv-python`
- `numpy`
- `scikit-learn`
- `seaborn`, `matplotlib`
- `tqdm`
- `jupyter`

> If you use SIFT in OpenCV ≥4.4, it’s included in `opencv-python`. For older builds, SIFT may require `opencv-contrib-python` instead.

---

## License
Choose a license (MIT/Apache‑2.0/BSD‑3‑Clause) and include a `LICENSE` file.

## Acknowledgements
- OpenCV community for feature detectors/descriptors
- scikit‑learn for KMeans/SVM and evaluation tools
- seaborn/matplotlib for visualization

---

**Maintainer tips**  
Clear cell outputs before committing to keep diffs small:
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "Phase1-Phase2-HandCraftedFeatures.ipynb"
```
Pin versions in `requirements.txt` for stable builds.
