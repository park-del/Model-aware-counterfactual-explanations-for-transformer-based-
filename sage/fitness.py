"""Confidence-aware semantic fitness from Eq. (21)."""

import numpy as np


def cosine_similarity(a, b, epsilon=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), epsilon)
    return float(np.dot(a, b) / denominator)


def semantic_similarity(candidate_static, candidate_dynamic, original_static, original_dynamic, dynamic_weight=0.7):
    static_weight = 1.0 - dynamic_weight
    return dynamic_weight * cosine_similarity(candidate_dynamic, original_dynamic) + static_weight * cosine_similarity(
        candidate_static, original_static
    )


def compute_fitness(target_probability, similarity, identical, alpha=0.5, beta=0.5, gamma=1.0):
    """Return alpha*p(target|z) + beta*Sim(z,sigma) - gamma*I(z=sigma)."""
    return alpha * float(target_probability) + beta * float(similarity) - gamma * float(bool(identical))
