"""Model checkpoint loading helpers."""

from pathlib import Path

import torch

from models.transformer import SequenceTransformer


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SequenceTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint

