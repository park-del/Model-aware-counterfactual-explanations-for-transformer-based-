"""Tokenizer for heterogeneous static attributes and dynamic activities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from preprocessing.common import CaseRecord


class ProcessTokenizer:
    PAD = "[PAD]"
    CLS = "[CLS]"
    UNK = "[UNK]"

    def __init__(self, max_length: int = 200, numeric_bins: int = 10):
        self.max_length = max_length
        self.numeric_bins = numeric_bins
        self.vocabulary = {self.PAD: 0, self.CLS: 1, self.UNK: 2}
        self.numerical_features: list[str] = []
        self.categorical_features: list[str] = []
        self.numeric_edges: dict[str, list[float]] = {}
        self.label_to_id: dict[str, int] = {}

    def _add(self, token: str) -> None:
        if token not in self.vocabulary:
            self.vocabulary[token] = len(self.vocabulary)

    def fit(self, cases: Sequence[CaseRecord]) -> "ProcessTokenizer":
        self.numerical_features = sorted({key for case in cases for key in case.numerical})
        self.categorical_features = sorted({key for case in cases for key in case.categorical})
        self.label_to_id = {label: index for index, label in enumerate(sorted({case.label for case in cases}))}
        for key in self.numerical_features:
            values = np.asarray(
                [case.numerical.get(key) for case in cases if case.numerical.get(key) is not None], dtype=float
            )
            if values.size:
                quantiles = np.linspace(0, 1, self.numeric_bins + 1)[1:-1]
                self.numeric_edges[key] = np.unique(np.quantile(values, quantiles)).astype(float).tolist()
            else:
                self.numeric_edges[key] = []
            self._add(f"NUM:{key}=<missing>")
            for bin_id in range(len(self.numeric_edges[key]) + 1):
                self._add(f"NUM:{key}=bin{bin_id}")
        for case in cases:
            for key in self.categorical_features:
                self._add(f"CAT:{key}={case.categorical.get(key, '<missing>')}")
            for activity in case.activities:
                self._add(f"ACT:{activity}")
        return self

    def _numeric_token(self, key: str, value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return f"NUM:{key}=<missing>"
        bin_id = int(np.searchsorted(self.numeric_edges[key], float(value), side="right"))
        return f"NUM:{key}=bin{bin_id}"

    def static_tokens(self, case: CaseRecord) -> list[str]:
        numerical = [self._numeric_token(key, case.numerical.get(key)) for key in self.numerical_features]
        categorical = [f"CAT:{key}={case.categorical.get(key, '<missing>')}" for key in self.categorical_features]
        return numerical + categorical

    def encode(self, case: CaseRecord, prefix_length: int) -> dict[str, list[int] | int]:
        static = self.static_tokens(case)
        dynamic = [f"ACT:{activity}" for activity in case.activities[:prefix_length]]
        tokens = [self.CLS] + static + dynamic
        type_ids = [0] + [1] * len(static) + [2] * len(dynamic)
        position_ids = [0] * (1 + len(static)) + list(range(1, len(dynamic) + 1))
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        position_ids = position_ids[: self.max_length]
        attention_mask = [1] * len(tokens)
        pad = self.max_length - len(tokens)
        tokens += [self.PAD] * pad
        type_ids += [0] * pad
        position_ids += [0] * pad
        attention_mask += [0] * pad
        return {
            "input_ids": [self.vocabulary.get(token, self.vocabulary[self.UNK]) for token in tokens],
            "type_ids": type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "label": self.label_to_id[case.label],
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "max_length": self.max_length,
            "numeric_bins": self.numeric_bins,
            "vocabulary": self.vocabulary,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "numeric_edges": self.numeric_edges,
            "label_to_id": self.label_to_id,
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProcessTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload["max_length"], payload["numeric_bins"])
        for key in ("vocabulary", "numerical_features", "categorical_features", "numeric_edges", "label_to_id"):
            setattr(tokenizer, key, payload[key])
        return tokenizer

    @property
    def id_to_label(self) -> dict[int, str]:
        return {value: key for key, value in self.label_to_id.items()}

