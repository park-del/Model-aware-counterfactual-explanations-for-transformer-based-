"""Semantic-aware initial population selection (Algorithm 1)."""

import numpy as np

from sage.fitness import cosine_similarity


def initialize_population(
    candidates,
    candidate_static,
    candidate_dynamic,
    original_static,
    original_dynamic,
    population_size,
    dynamic_weight=0.7,
):
    scores = np.asarray(
        [
            dynamic_weight * cosine_similarity(dynamic, original_dynamic)
            + (1.0 - dynamic_weight) * cosine_similarity(static, original_static)
            for static, dynamic in zip(candidate_static, candidate_dynamic)
        ]
    )
    order = np.argsort(-scores)[:population_size]
    return [candidates[index] for index in order], scores[order]
