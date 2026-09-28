"""Attention-guided, domain-valid mutation from Eqs. (25)-(26)."""

from __future__ import annotations

import numpy as np


def mutation_distribution(attention_scores, temperature=0.5):
    scores = np.asarray(attention_scores, dtype=float)
    scaled = scores / max(float(temperature), 1e-8)
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum()


def mutate(candidate, attention_scores, domains, tokenizer, rng, temperature=0.5):
    """Mutate one encoded position and draw a replacement from its observed domain."""
    static_keys = [("numerical", key) for key in tokenizer.numerical_features] + [
        ("categorical", key) for key in tokenizer.categorical_features
    ]
    mutable = list(range(1, 1 + len(static_keys) + len(candidate.activities)))
    probabilities = mutation_distribution([attention_scores[index] for index in mutable], temperature)
    position = rng.choices(mutable, weights=probabilities.tolist(), k=1)[0]
    static_offset = position - 1
    if static_offset < len(static_keys):
        block, key = static_keys[static_offset]
        values = [value for value in domains[block][key] if value != getattr(candidate, block).get(key)]
        if values:
            getattr(candidate, block)[key] = rng.choice(values)
        return candidate
    activity_position = static_offset - len(static_keys)
    values = [
        value
        for value in domains["activities"].get(activity_position, domains["all_activities"])
        if value != candidate.activities[activity_position]
    ]
    if values:
        candidate.activities[activity_position] = rng.choice(values)
    return candidate
