# Model-Aware Counterfactual Explanations for Transformer-Based Sequence Prediction

Reproduction repository for the paper **Model-Aware Counterfactual Explanations for Transformer-Based Sequence Prediction via Attention-Guided Semantic Search**.

## Datasets

| Dataset | Raw file | Task |
|---|---|---|
| BPIC 2017 | `BPI Challenge 2017.xes.gz` | Three-class prediction |
| BPIC 2012 | `BPI Challenge 2012.xes.gz` | Three-class prediction |
| Helpdesk | `finale.csv` | Binary prediction |
| Sepsis | `Sepsis Cases - Event Log.xes.gz` | Binary prediction |

Download instructions are in [data/README.md](data/README.md). Raw datasets, checkpoints, and experiment outputs are intentionally excluded from Git.

## Repository structure

- `preprocessing/`: dataset parsing, prefix construction, feature encoding, and splitting
- `models/`: Transformer-based sequence predictor
- `sage/`: attention-guided semantic search and counterfactual generation
- `baselines/`: comparison methods
- `experiments/`: training, counterfactual generation, ablation, and evaluation scripts
- `utils/`: configuration, metrics, reproducibility, and shared helpers
- `configs/`: experiment configuration files
- `data/`: local raw and processed datasets

## Current status

The repository skeleton is ready. Implement the pipeline in this order:

1. Dataset preprocessing
2. Transformer predictor
3. SAGE counterfactual generator
4. Baselines and evaluation
5. Ablation and sensitivity experiments
