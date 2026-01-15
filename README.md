# Towards Best Practices of Axiomatic Activation Patching in Information Retrieval

This repository hosts the code for reproducing the experiments presented in our paper:

> **Towards Best Practices of Axiomatic Activation Patching in Information Retrieval**  
> Gregory Polyakov, Catherine Chen, Carsten Eickhoff  
> *SIGIR 2025*  
> [https://dl.acm.org/doi/10.1145/3726302.3730256](https://dl.acm.org/doi/10.1145/3726302.3730256)

Our code is based on the [MechIR library](https://github.com/Parry-Parry/MechIR) by Parry et al. (ECIR 2025).

---

## Installation

This project requires **Python 3.9**.

Install the main dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
|-- data/
|   |-- diagnostic_dataset/
|   |-- patching_results/
|-- notebooks/
|   |-- cross_encoder_correlation_analysis_TFC1.ipynb
|   |-- dual_encoder_correlation_analysis_TFC1.ipynb
|-- README.md
|-- requirements.txt
|-- scripts/
|   |-- collect_documents.py
|   |-- collect_queries.py
|   |-- generate_features.py
|   |-- patch.py
|   |-- run_patch.sh
|-- utils/
    |-- __pycache__/
    |   |-- utils.cpython-311.pyc
    |-- utils.py
```

---

## Data Generation

All data used in our experiments is available here: **[Google Drive link](https://drive.google.com/drive/folders/19MkcC6KVLGAOIu5H9pPyayu7o5tASpTZ?usp=share_link)**

You could download the necessary files and put them into the `data/` directory.

To generate features of queries and documents used in our correlation analyses, run:

```bash
python scripts/generate_features.py
```

---

## Main Experiments

All main experiments for Dual Encoder and Cross Encoder models are located in the following notebooks:

* **`notebooks/cross_encoder_correlation_analysis_TFC1.ipynb`**
* **`notebooks/dual_encoder_correlation_analysis_TFC1.ipynb`**

These notebooks contain the complete workflow for the experiments described in the paper, including:

* Visualization of Patching Effects
* Correlation Analysis
* CatBoost Feature Importance Analysis

---

