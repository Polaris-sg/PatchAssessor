# Enhancing Large Language Models for Patch Correctness Assessment via Coarse-to-Fine Retrieval

## Overview

This project provides the implementation of a **coarse-to-fine retrieval framework** designed to enhance **large language models (LLMs)** for **patch correctness assessment **.
The framework supports patch data processing, similarity-based retrieval, correctness prediction using LLMs, and comprehensive evaluation.

## Key Features

1. **Data Processing**
   Loads patch data from `PCA_Func_dataset.json` and divides it into query and repository sets according to the specified repair tool name.

2. **Patch Retrieval**
   Retrieves similar patches from the repository using multiple similarity computation methods.
   Three retrieval modes are supported:

   * **coarse**: Text-based similarity comparing changed lines (lines starting with `+` or `-`) and their context.
   * **fine**: Graph-based similarity converting patches into structural graph representations for comparison.
   * **coarse2fine**: Two-stage retrieval — first performs coarse filtering via text similarity, then refines the ranking using graph similarity.

3. **Patch Correctness Prediction**
   Employs large language models to predict whether a given query patch is correct based on its retrieved similar patches.

4. **Result Evaluation**
   Computes standard evaluation metrics including Accuracy, F1-score, and AUC.

> **Note:**
> In the dataset, patch labels follow this convention:
> **0 → Correct patch**, **1 → Incorrect patch**.

---

## Installation

Ensure that all required dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## Main Modules

* `data_processing.py`: Processes the `apca_patch_summary.json` dataset.
* `patch_retrieval.py`: Implements patch similarity retrieval.
* `patch_predictor.py`: Predicts patch correctness using LLMs.
* `evaluation.py`: Calculates evaluation metrics.

---

## Usage

### 1. Data Processing

```bash
python data_processing.py --tool_name Arja --output_dir ./results/processed_data
```

### 2. Patch Retrieval

```bash
python patch_retrieval.py \
  --query_path ./processed_data/Arja_query_set.jsonl \
  --repository_path ./processed_data/Arja_repository_set.jsonl \
  --output_path ./results/search_results \
  --mode coarse2fine \
  --gamma 0.1
```

### 3. Patch Correctness Prediction

```bash
python patch_predictor.py \
  --input_path ./search_results/search_results.jsonl \
  --output_path ./prediction_results/prediction_results.jsonl
```

Using different models:

```bash
# Use the CodeLlama model
python patch_predictor0917.py \
  --input_path ./results/search_results \
  --output_path ./results/prediction_results \
  --model codellama-7b-hf

# Specify the number of retrieved similar patches
python patch_predictor.py \
  --input_path ./search_results\
  --output_path ./prediction_results \
  --top_k 10
```

### 4. Evaluation

```bash
python evaluation.py \
  --predictions_path ./results/prediction_results\
  --output_path ./results/evaluation_results
```

---

Would you like me to help you extend this README with a short **“Citation”** and **“License”** section (for use in your paper’s replication package)? I can make it ready for GitHub directly (including Markdown formatting and citation entry).
