# ICU Time Series Analysis: Predictive Modeling and Representation Learning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Abstract

Intensive care patients, usually being severely ill, are closely monitored by state of the art
medical devices and sensors to measure vital signs at regular intervals. Additionally, doctors
take frequent laboratory tests and any type of treatment administered via infusion, injection,
or oxygen supplied through various ventilation equipment, are all meticulously recorded.

These large amounts of data can be overwhelming for intensive care physicians to analyze
properly while under time pressure. Machine learning algorithms are prime candidates to
leverage these large amounts of data and distill them into actionable outputs for clinical
decision support.

The goal of this project is to get familiar with such a promising type of healthcare data, while
at the same time learning how to tackle the challenges of working on noisy, multi-variate,
irregularly-sampled and sparse, and confounded real world data.

## 📦 Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   ├── data_1         <- Contains the initial raw data from the challenge.
│   └── processed      <- The final, cleaned data used for modeling.
│
├── docs               <- Documentation files for the project
│   ├── Project 1 - ICU Time Series - Handout.pdf
│   └── Project 1 - ICU Time Series - Slides.pdf
│
├── models             <- Trained models, predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         project_1 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and other reference materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated figures for reporting and presentations
│
├── requirements.txt   <- The requirements file to reproduce the analysis environment
│
├── setup.cfg          <- Configuration file for linting tools like flake8
│
├── scripts            <- Shell scripts or utilities for automation
│   └── run_Q1_data_processing.sh  <- Script to execute all Q1 processing steps
│
└── project_1          <- Python module containing the core logic for the project
    ├── __init__.py
    ├── Q1/            <- Code for Question 1 (e.g., preprocessing, cleaning)
    ├── Q2/            <- Code for Question 2 (e.g., tokenization, embeddings)
    ├── Q4/            <- Code for Question 4 (e.g., classifier training)
    ├── loading.py     <- Functions to load raw and processed data
    ├── features.py    <- Functions for feature scaling and preprocessing
    └── dataset.py     <- Functions to convert data into PyTorch datasets

```

---

## ⚙️ Setup

Install dependencies (recommended via virtual environment):

```
pip install -r requirements.txt
```

### ⚠️ Ollama Note

This project uses the ollama Python package to interface with locally running language models. Please ensure that Ollama is installed on your device before using this feature—otherwise, the Python package will not function correctly.
You can follow the installation guide here: https://ollama.com/download

---

## 🚀 How to Run the Project

This project is organized into four main questions (Q1–Q4), each addressing a different part of ICU time-series analysis. Follow the instructions below to reproduce the results.

---

### 📊 Q1: Data Preprocessing

1. Download the challenge dataset from PhysioNet:  
   [`predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0`](https://physionet.org/content/challenge-2012/1.0.0/)

2. Place the extracted folder inside:

   ```
   data/data_1/
   ```

3. Run the preprocessing script to clean and process the raw data:

   ```
   bash scripts/run_Q1_data_processing.sh
   ```

   This will generate all necessary `.parquet` files inside:

   ```
   data/processed/
   ```

---

### 🤖 Q2: Supervised Learning

Explore different supervised learning models and results in the following notebook:

📓 `notebooks/2_Q2_supervised_learning.ipynb`

---

### 🤯 Q3: Representation Learning

Dive into self-supervised learning and contrastive representation learning in:

📓 `notebooks/3_Q3_representation_learning.ipynb`

---

### 🧠 Q4: Foundation Models

Analyze foundation model performance and transfer learning setups in:

📓 `notebooks/4_Q4_foundation_models.ipynb`
