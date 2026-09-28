"""PyTorch prefix dataset for processed event-log cases."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from models.tokenizer import ProcessTokenizer
from preprocessing.common import CaseRecord, iter_prefix_lengths


class PrefixDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseRecord],
        tokenizer: ProcessTokenizer,
        minimum_prefix_length: int = 1,
        prefix_step: int = 1,
    ):
        self.cases = list(cases)
        self.tokenizer = tokenizer
        static_count = 1 + len(tokenizer.numerical_features) + len(tokenizer.categorical_features)
        maximum_dynamic = max(1, tokenizer.max_length - static_count)
        self.index = [
            (case_index, length)
            for case_index, case in enumerate(self.cases)
            for length in iter_prefix_lengths(case, minimum_prefix_length, maximum_dynamic, prefix_step)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        case_index, prefix_length = self.index[index]
        encoded = self.tokenizer.encode(self.cases[case_index], prefix_length)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}

