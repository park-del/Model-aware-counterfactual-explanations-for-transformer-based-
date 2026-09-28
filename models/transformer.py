"""Transformer encoder with layer/head attention export for SAGE."""

from __future__ import annotations

import torch
from torch import nn


class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, hidden, key_padding_mask):
        attended, weights = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = self.norm1(hidden + self.dropout(attended))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(hidden))))
        hidden = self.norm2(hidden + self.dropout(feedforward))
        return hidden, weights


class SequenceTransformer(nn.Module):
    """Unified-token Transformer matching the paper's 3-layer/8-head defaults."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        max_len: int = 200,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "max_len": max_len,
            "feedforward_dim": feedforward_dim,
            "dropout": dropout,
        }
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(3, d_model)
        self.position_embedding = nn.Embedding(max_len + 1, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [AttentionEncoderLayer(d_model, nhead, feedforward_dim, dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, type_ids, position_ids, attention_mask, return_attention=False):
        hidden = (
            self.token_embedding(input_ids)
            + self.type_embedding(type_ids)
            + self.position_embedding(position_ids)
        )
        hidden = self.dropout(self.embedding_norm(hidden))
        key_padding_mask = ~attention_mask.bool()
        attentions = []
        for layer in self.layers:
            hidden, weights = layer(hidden, key_padding_mask)
            if return_attention:
                attentions.append(weights)
        pooled = hidden[:, 0]
        result = {"logits": self.classifier(pooled), "embedding": pooled}
        if return_attention:
            result["hidden_states"] = hidden
            result["attentions"] = torch.stack(attentions, dim=0)
            result["aggregated_attention"] = result["attentions"].mean(dim=(0, 2))
        return result

    @torch.no_grad()
    def probabilities(self, batch):
        return self(**batch)["logits"].softmax(dim=-1)
