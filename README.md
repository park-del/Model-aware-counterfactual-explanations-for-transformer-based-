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

## Quick start

```bash
pip install -r requirements.txt
python main.py --stage download
python main.py --stage preprocess --dataset sepsis
python main.py --stage train --dataset sepsis
python main.py --stage counterfactual --dataset sepsis
python main.py --stage evaluate --dataset sepsis
```

The data manifest pins official downloads and SHA-256 checksums. The preprocessing pipeline, attention-exporting Transformer, SAGE evolutionary search, local surrogate rules, and metric aggregation are implemented. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before comparing numbers with the paper: several label and field-mapping details are not specified in the article and are exposed as explicit assumptions here.
