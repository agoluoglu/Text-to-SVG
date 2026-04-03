# Text-to-SVG Generation via LoRA Fine-Tuning

NYU Tandon Deep Learning (ECE-GY 7123), Spring 2026  
Kaggle Competition: Text-to-SVG Generation from Natural Language Prompts
 
**Team:** WorkinTilTheAM (Michael Blanchette, Ashley Goluoğlu)
 
**Public Leaderboard:** 11.17 (Rank 77/89) (rip lol)
 
## Overview

LoRA fine-tuning of Qwen2.5-3B-Instruct for text-to-SVG generation. Aggressive data preprocessing (coordinate rounding, attribute cleaning, outlier removal) to maximize token efficiency, followed by multi-stage inference with retry/fallback logic to guarantee 100% valid submissions.
 
## Repository Structure
 
```
├── notebooks/
│   ├── 01_data_exploration.ipynb                   # Data analysis + preprocessing pipeline
│   ├── 02_train_and_inference_ColabA100.ipynb      # Training + inference (Colab A100)
│   └── 02-train-and-inference_kaggle.ipynb         # Training + inference (Kaggle T4)
├── requirements.txt
├── NYU_DeepLearning_Spring2026_MidtermReport.pdf   # Report
└── README.md
```
 
## HuggingFace Resources
 
| Resource | Link |
|----------|------|
| Filtered training data | [aagoluoglu/text-to-svg](https://huggingface.co/datasets/aagoluoglu/text-to-svg) (`train_filtered.csv`) |
| LoRA adapter weights | [aagoluoglu/qwen-svg-lora](https://huggingface.co/aagoluoglu/qwen-svg-lora) 
| Submission | [aagoluoglu/text-to-svg](https://huggingface.co/datasets/aagoluoglu/text-to-svg) (`submission.csv`) |
 
## How to Reproduce
 
### Step 1: Data Preprocessing
 
Run `01_data_exploration.ipynb` on any machine (no GPU needed). This notebook:
 
- Loads raw `train.csv` and `test.csv` from HuggingFace
- Rounds coordinates (precision customizable with DECIMALS variable), removes duplicates, cleans attributes, filters outliers
- Exports `train_filtered.csv` to HuggingFace at `aagoluoglu/text-to-svg`
 
If you want to use our already-filtered data, skip this step!
 
### Step 2: Training + Inference
 
Pick the right notebook for your environment:
 
| Notebook | GPU | Use When |
|----------|-----|----------|
| `02_train_and_inference_ColabA100.ipynb` | A100 (40GB) | Full training + fast inference |
| `02-train-and-inference_kaggle.ipynb` | T4 (16GB) | Kaggle submission (required for competition) |
 
The kaggle notebook works on Colab T4 as well without changes (auto-detects and adjusts accordingly), but the A100 notebook contains adjustments to take advantage of the increased capabilities, and will not work on kaggle without modifications (as kaggle does not have A100). 
 
**Key differences between the two notebooks:**
 
- **A100 version:** Loads merged model in bf16 for inference (no Unsloth dependency at inference time). Uses `torch.inference_mode()` and batch size 64. Has `# EDITS for A100` comments marking all changes from the base version.
- **Kaggle version:** Uses Unsloth's `FastLanguageModel` for both training and inference in 4-bit mode. Batch size 8. This is the version that runs end-to-end in Kaggle's "Save & Run All" format.
 
**To train from scratch:**
 
1. Set `DO_TRAINING = True` at the top of the notebook. Make sure `TESTING = False` (or `True` if you want to test functionality on very small sample).
2. Set your HuggingFace token in the secrets/environment
3. Run all cells (training takes ~6-8 hours on A100)
 
**To run inference only (using our pre-trained adapter):**
 
1. Set `DO_TRAINING = False`. Make sure `TESTING = False` (or `True` if you want to test functionality on very small sample).
2. The notebook will pull the adapter from `aagoluoglu/qwen-svg-lora`
3. Run all cells (inference takes ~3 hours on A100, much longer on T4)
 
### Final Config
 
```
Model:            Qwen2.5-3B-Instruct (4-bit quantized via Unsloth)
LoRA rank:        64
LoRA alpha:       64
Target modules:   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Epochs:           2
Learning rate:    2e-5
Effective batch:  16 (batch=2, accumulation=8)
max_seq_length:   2048
Seed:             42
```
 
## Requirements
 
Both notebooks install their own dependencies. Core packages:
 
- `unsloth`
- `transformers`
- `peft`
- `datasets`
- `cairosvg`
- `torch`

Check `requirements.txt` for full list.
Note: pin to transformers version 4.56.2 to fix a shape mismatch bug that unsloth has.
 
## AI Tooling
 
Claude (Anthropic) was used as a coding/debugging assistant. All architecture and experimental decisions were made by the team.


## Results
| Split | Score |
|-------|-------|
| Public leaderboard | 11.17402 |
| Private leaderboard | TBD |
