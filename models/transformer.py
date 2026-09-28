"""Transformer encoder for sequence outcome prediction.

Paper defaults:
- encoder layers: 3
- attention heads: 8
- hidden dimension: 128
- maximum sequence length: 200
"""

from torch import nn


class SequenceTransformer(nn.Module):
    """Transformer-based sequence classifier scaffold."""

    def __init__(
        self,
        num_activities,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=3,
        max_len=200,
    ):
        super().__init__()
        self.num_activities = num_activities
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        raise NotImplementedError("The predictor architecture will be implemented next.")

    def forward(self, batch):
        """Return class logits and attention information."""
        raise NotImplementedError
