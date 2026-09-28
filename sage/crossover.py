"""Structure-aware crossover that never mixes static and dynamic blocks."""

from copy import deepcopy


def crossover(parent_a, parent_b, rng):
    child_a, child_b = deepcopy(parent_a), deepcopy(parent_b)
    for block in ("numerical", "categorical"):
        values_a = getattr(child_a, block)
        values_b = getattr(child_b, block)
        for key in sorted(set(values_a) & set(values_b)):
            if rng.random() < 0.5:
                values_a[key], values_b[key] = values_b[key], values_a[key]
    length = min(len(child_a.activities), len(child_b.activities))
    if length > 1:
        start = rng.randrange(length)
        end = rng.randrange(start + 1, length + 1)
        segment_a = child_a.activities[start:end]
        child_a.activities[start:end] = child_b.activities[start:end]
        child_b.activities[start:end] = segment_a
    return child_a, child_b
