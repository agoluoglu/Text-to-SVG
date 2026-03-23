# DL Spring 2026 — Text-to-SVG Generation

NYU Tandon Deep Learning (ECE-GY 7123), Spring 2026  
Kaggle Competition: Text-to-SVG Generation from Natural Language Prompts

## Overview
Fine-tuned Qwen2.5-1.5B-Instruct with QLoRA (4-bit) to generate valid SVG code
from text prompts. Trained on the competition's 50,000 prompt-SVG pairs.

## Repository Structure
```
notebooks/
  01_data_exploration.ipynb   # EDA, length analysis, filtering decisions
  02_training.ipynb           # QLoRA fine-tuning on Colab T4
  03_kaggle_inference.ipynb   # Kaggle submission notebook
requirements.txt
```

## Reproducing Results

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Data
Download `train.csv` and `test.csv` from the
[Kaggle competition page](https://www.kaggle.com/competitions/dl-spring-2026-svg-generation/data).  
Training data is also mirrored at:
`https://huggingface.co/datasets/YOUR_HF_USERNAME/svg-train` (private)

### 3. Training
Open `notebooks/02_training.ipynb` in Google Colab (use T4 GPU).  
Set your HuggingFace token in Colab secrets as `HF_TOKEN`.

### 4. Inference
Open `notebooks/03_kaggle_inference.ipynb` on Kaggle.  
Attach the model weights dataset before running.

## Model Weights
Trained LoRA adapter: [HuggingFace Hub](https://huggingface.co/YOUR_HF_USERNAME/svg-lora-adapter)

## Results
| Split | Score |
|-------|-------|
| Public leaderboard | TBD |
| Private leaderboard | TBD |

## Dependencies
See `requirements.txt`. Key packages: `unsloth`, `transformers`, `trl`, `peft`.
