# project-1

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
        ├── set_a      
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         project_1 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── project_1   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes project_1 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

## How to use:

In the `scripts/` folder there are various `.sh` files. Simply run `./scripts/[name].sh` to execute the whole processes.
