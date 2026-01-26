# Brain Tumor Classification

The objective of this project is to propose a pipeline and a model that can classify brain images into four labels:
- glioma
- meningioma
- pituitary
- no tumor

---

## 1. Data

### 1.1 Sources

Dataset : Figshare Brain Tumor Dataset  
Link : https://www.kaggle.com/datasets/denizkavi1/brain-tumor/data

Dataset :  Brain Tumor MRI Dataset
Link : https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset : Multi-Cancer Dataset (Brain Subset)
Link : https://www.kaggle.com/datasets/obulisainaren/multi-cancer

Dataset : Brain Tumor MRI Scans (Kaggle)
Link : https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans

### 1.2 Preprocessing
- resize
- normalization
- data augmentation
- split in train/validation/test

---

## 2. Structure of the project

```
BRAIN_TUMOR_CLASSIFICATION/
│
├── data/
│ ├── raw/
│ └── processed/
|
├── graphics/
│
├── notebooks/
│ ├── figures/
│ └── eda_dataset.ipynb
│
├── src/
│ ├── preprocessing/
│ │ └── preprocessing.py
│ └── Brain_Tumor_Classification.ipynb
│
├── venv/
│
├── .gitignore
├── README.md
└── requirements.txt
```
---

## 3. Installation

### 3.1 Prerequisites (Local - Preprocessing)
- Python 3.13
- Required libraries for preprocessing (listed in `requirements.txt`)

### 3.2 Create a virtual environment (Local)
```bash
python -m venv venv
```

### 3.3 Activate the virtual environment (Local)

Linux / macOS:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### 3.4 Install dependencies (Local)

```bash
pip install -r requirements.txt
```
---

## 4. Data Preprocessing Pipeline

This module implements a robust preprocessing pipeline designed to standardize heterogenous Brain Tumor MRI datasets before training. It handles data cleaning, normalization, artifact removal, and dataset splitting.

### 4.1 Key Features

The pipeline executes the following steps sequentially:

1.  **Color Standardization (3-Channel RGB)**
    * Converts all inputs (Grayscale, RGBA, etc.) to a standard 3-channel RGB format.
    * Detects "effective grayscale" images saved as RGB and normalizes them to ensure consistency across the dataset.

2.  **Content-Based Deduplication**
    * Uses a perceptual-like hashing technique: images are converted to grayscale and resized to a **32x32 thumbnail**.
    * An MD5 hash is computed on this thumbnail.
    * **Benefit:** This removes duplicates even if the files have slight color variations (e.g. one image is purple-tinted and the duplicate is grayscale).

3.  **Conditional Orientation Correction**
    * Detects images that are rotated 90° (landscape orientation where portrait is expected).
    * **Constraint:** This correction is applied *only* to filenames starting with `brain_` to avoid altering already correct datasets or specific classes.

4.  **ROI Extraction (Region of Interest)**
    * Automatically detects the brain contour using morphological operations (Gaussian Blur, Thresholding, Erosion/Dilation).
    * Crops the image to the bounding box of the brain, removing useless black backgrounds and reducing image dimensions.

5.  **Resize & Split**
    * Resizes the cropped ROI to a fixed size (default: `224x224`).
    * Performs a random **70% Train / 15% Val / 15% Test** uniform split across classes.

### 4.2 Running the pipeline

Run the script from the command line by specifying the input directory containing your raw data.

Basic usage:
```bash
python preprocessing.py --input path/to/raw_dataset
```

Custom output path and image size:
```bash
python preprocessing.py --input "../../data/raw/dataset" --output "../../data/processed/dataset_final" --size 224
```
---

## 5. Brain Tumor Classification

To fine-tune the models, you need to use this notebook:

```bash
src/Brain_Tumor_Classification.ipynb
```

You need to download it and open it directly on Google Colab in order to benefit from the computing power of this tool.