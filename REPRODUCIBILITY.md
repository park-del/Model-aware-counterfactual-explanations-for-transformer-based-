# Reproducibility notes

This repository separates facts stated in the published paper from assumptions required to turn the public raw logs into executable experiments.

## Fixed by the paper

- Datasets: BPIC 2017, BPIC 2012, Helpdesk, and Sepsis
- Case split target: 72% training, 8% validation, 20% test
- Transformer: 3 encoder layers, 8 heads, embedding dimension 128, maximum sequence length 200
- Training: categorical cross-entropy, Adam, learning rate `1e-4`, batch size 128, at most 50 epochs, validation-loss early stopping
- SAGE: population 1000, 10 generations, crossover probability 0.7, mutation probability 0.2
- Surrogate tree maximum depth: 5

## Information not published in sufficient detail

1. BPIC 2017 contains 98 traces without `A_Pending`, `A_Denied`, or `A_Cancelled`; BPIC 2012 contains 399 traces without `A_APPROVED`, `A_DECLINED`, or `A_CANCELLED`. The paper reports all raw cases but does not state how these traces receive one of three labels. The implementation excludes them and records the resulting counts in `metadata.json`.
2. Helpdesk is described as timely versus delayed according to total resolution time, but the threshold is not reported. The default is the median duration and can be overridden with `timely_threshold_hours`.
3. The paper reports static feature counts but not the complete field mapping or aggregation rules for BPIC 2012, Helpdesk, and Sepsis. The implementation retains documented raw case features and records both actual and paper-reported feature lists.
4. Prefix sampling during predictor training, random seeds for reported repeated runs, baseline library versions, and exact baseline hyperparameters are not fully specified.

These gaps prevent a defensible claim of bit-for-bit or table-for-table reproduction without the authors' original experiment configuration. The code therefore emits every assumption instead of silently inventing it.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --stage download
python main.py --stage preprocess --dataset sepsis
python main.py --stage train --dataset sepsis
python main.py --stage counterfactual --dataset sepsis
python main.py --stage evaluate --dataset sepsis
```

Repeat the last four commands for `helpdesk`, `bpic2012`, and `bpic2017`.

Baseline, ablation, and sensitivity commands are documented in `README.md`. Baseline modules are transparent Table-7-compatible reimplementations because the article does not publish the authors' exact baseline source tree.

The paper used Python 3.10, PyTorch 2.1.2, CUDA 12.1, Ubuntu 22.04, and one RTX 4090 GPU.
