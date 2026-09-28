"""Train and evaluate the Transformer sequence predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.data import PrefixDataset
from models.tokenizer import ProcessTokenizer
from models.transformer import SequenceTransformer
from preprocessing.common import read_jsonl
from utils.config import load_config
from utils.seed import set_seed


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses, labels, probabilities = [], [], []
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        batch = move_batch(batch, device)
        label = batch.pop("label")
        logits = model(**batch)["logits"]
        losses.append(criterion(logits, label).item() * label.numel())
        labels.extend(label.cpu().tolist())
        probabilities.extend(logits.softmax(dim=-1).cpu().tolist())
    probability_array = np.asarray(probabilities)
    predictions = probability_array.argmax(axis=1)
    metrics = {
        "loss": sum(losses) / max(1, len(labels)),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }
    try:
        metrics["macro_auc"] = roc_auc_score(
            labels,
            probability_array if probability_array.shape[1] > 2 else probability_array[:, 1],
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        metrics["macro_auc"] = None
    return metrics


def run(config, dataset):
    set_seed(int(config["project"]["seed"]))
    processed = ROOT / config["data"]["processed_dir"] / dataset
    output = ROOT / config["project"]["output_dir"] / dataset
    output.mkdir(parents=True, exist_ok=True)
    train_cases = read_jsonl(processed / "train.jsonl")
    validation_cases = read_jsonl(processed / "validation.jsonl")
    test_cases = read_jsonl(processed / "test.jsonl")

    tokenizer = ProcessTokenizer(max_length=int(config["data"]["max_sequence_length"]))
    tokenizer.fit(train_cases)
    tokenizer.save(output / "tokenizer.json")
    dataset_options = {
        "minimum_prefix_length": int(config["data"]["minimum_prefix_length"]),
        "prefix_step": int(config["data"]["prefix_step"]),
    }
    datasets = {
        "train": PrefixDataset(train_cases, tokenizer, **dataset_options),
        "validation": PrefixDataset(validation_cases, tokenizer, **dataset_options),
        "test": PrefixDataset(test_cases, tokenizer, **dataset_options),
    }
    batch_size = int(config["model"]["batch_size"])
    loaders = {
        name: DataLoader(value, batch_size=batch_size, shuffle=name == "train", num_workers=0)
        for name, value in datasets.items()
    }

    model_config = {
        "vocab_size": len(tokenizer.vocabulary),
        "num_classes": len(tokenizer.label_to_id),
        "d_model": int(config["model"]["d_model"]),
        "nhead": int(config["model"]["num_heads"]),
        "num_layers": int(config["model"]["num_layers"]),
        "max_len": int(config["data"]["max_sequence_length"]),
        "feedforward_dim": int(config["model"]["feedforward_dim"]),
        "dropout": float(config["model"]["dropout"]),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceTransformer(**model_config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    patience = int(config["model"]["early_stopping_patience"])
    best_loss, stale = float("inf"), 0
    history = []
    for epoch in range(1, int(config["model"]["max_epochs"]) + 1):
        model.train()
        total_loss = total_items = 0
        progress = tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            label = batch.pop("label")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch)["logits"]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * label.numel()
            total_items += label.numel()
            progress.set_postfix(loss=total_loss / total_items)
        validation = evaluate(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": total_loss / total_items, **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(row)
        print(json.dumps(row))
        if validation["loss"] < best_loss:
            best_loss, stale = validation["loss"], 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "dataset": dataset,
                    "tokenizer": str(output / "tokenizer.json"),
                    "epoch": epoch,
                    "validation": validation,
                },
                output / "model.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(output / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, loaders["test"], device)
    result = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["validation"], "test": test_metrics, "history": history}
    (output / "training_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["test"], indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=("bpic2017", "bpic2012", "helpdesk", "sepsis"))
    args = parser.parse_args()
    run(load_config(ROOT / args.config), args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
