# Enhancing Large Language Models for Patch Correctness Assessment via Coarse-to-Fine Retrieval

## Overview

In this paper, we propose PatchAssessor, a patch assessment framework that enhances the reasoning capability of LLM for patch correctness via a coarse-to-fine retrieval strategy. PatchAssessor first partitions the historical patch repository and query patches on our constructed patch dataset PCA-Func, then employs a two-stage retrieval combining textual and graph-based structural similarity to identify relevant patches. Retrieved patches, together with query patch, bug, and test information, are formulated into structured prompts to guide the LLM in accurate patch assessment.

## Key Features

1. **Data Processing**
   Loads patch data from `PCA_Func_dataset.json` and divides it into query and repository sets according to the specified repair tool name.

2. **Patch Retrieval**
   Retrieves similar patches from the repository using multiple similarity computation methods.
   
4. **Patch Correctness Prediction**
   Employs large language models to predict whether a given query patch is correct based on its retrieved similar patches.

5. **Result Evaluation**
   Computes standard evaluation metrics including Accuracy, F1-score, and AUC.

---

## Installation

Ensure that all required dependencies are installed:

```bash
pip install -r requirements.txt
```

---

## Main Modules

* `data_processing.py`: Processes the `PCA_Func_dataset.json` dataset.
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
python patch_retrieval.py 
  --query_path ./processed_data/Arja_query_set.jsonl 
  --repository_path ./processed_data/Arja_repository_set.jsonl 
  --output_path ./results/search_results 
  --mode coarse2fine 
  --gamma 0.1
```

### 3. Patch Correctness Prediction

```bash
python patch_predictor.py --input_path ./search_results/search_results.jsonl --output_path ./prediction_results/prediction_results.jsonl
```

Using different models:

```bash
# Use the CodeLlama model
python patch_predictor0917.py --input_path ./results/search_results --output_path ./results/prediction_results --model codellama-7b-hf

# Specify the number of retrieved similar patches
python patch_predictor.py \ --input_path ./search_results\ --output_path ./prediction_results \ --top_k 5
```

### 4. Evaluation

```bash
python evaluation.py --predictions_path ./results/prediction_results --output_path ./results/evaluation_results
```



