"""Aggregate the counterfactual and surrogate metrics reported by SAGE."""

from __future__ import annotations

import math
import statistics

from preprocessing.common import CaseRecord


def _summary(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for row_index, left_value in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def numeric_ranges(reference_cases):
    ranges = {}
    keys = {key for case in reference_cases for key in case.numerical}
    for key in keys:
        values = [case.numerical.get(key) for case in reference_cases if case.numerical.get(key) is not None]
        ranges[key] = (min(values), max(values)) if values else (0.0, 0.0)
    return ranges


def gower_distance(left, right, ranges):
    components = []
    for key in sorted(set(left.numerical) | set(right.numerical)):
        a, b = left.numerical.get(key), right.numerical.get(key)
        if a is None or b is None:
            components.append(float(a != b))
        else:
            minimum, maximum = ranges.get(key, (a, b))
            components.append(abs(float(a) - float(b)) / max(float(maximum) - float(minimum), 1e-12))
    for key in sorted(set(left.categorical) | set(right.categorical)):
        components.append(float(left.categorical.get(key) != right.categorical.get(key)))
    return statistics.fmean(components) if components else 0.0


def mixed_distance(left, right, ranges, static_weight=0.5):
    dynamic = levenshtein(left.activities, right.activities) / max(1, len(left.activities), len(right.activities))
    return static_weight * gower_distance(left, right, ranges) + (1.0 - static_weight) * dynamic


def sparsity(left, right):
    static = sum(left.numerical.get(key) != right.numerical.get(key) for key in set(left.numerical) | set(right.numerical))
    static += sum(left.categorical.get(key) != right.categorical.get(key) for key in set(left.categorical) | set(right.categorical))
    return static + levenshtein(left.activities, right.activities)


def evaluate_counterfactuals(results, reference_cases):
    ranges = numeric_ranges(reference_cases)
    top = [result["counterfactuals"][0] for result in results if result.get("counterfactuals")]
    successful = [result for result in results if result.get("counterfactuals")]
    pairs = [
        (CaseRecord(**result["original"]), CaseRecord(**item["case"]), result["target_class"])
        for result, item in zip(successful, top)
    ]
    syntactic = [1.0 - mixed_distance(original, candidate, ranges) for original, candidate, _ in pairs]
    plausibility = []
    for _, candidate, target in pairs:
        distances = sorted(
            mixed_distance(candidate, reference, ranges)
            for reference in reference_cases
            if reference.label == target
        )[:5]
        plausibility.append(1.0 / (1.0 + statistics.fmean(distances) + 1e-12) if distances else None)
    rules = [rule for result in results for rule in result.get("surrogate", {}).get("rules", [])]
    return {
        "instances": len(results),
        "validity": sum(bool(result.get("validity")) for result in results) / max(1, len(results)),
        "sparsity": _summary([sparsity(original, candidate) for original, candidate, _ in pairs]),
        "syntactic_proximity": _summary(syntactic),
        "semantic_proximity": _summary([item["semantic_similarity"] for item in top]),
        "plausibility": _summary(plausibility),
        "runtime_seconds": _summary([result.get("runtime_seconds") for result in results]),
        "surrogate_fidelity": _summary([result.get("surrogate", {}).get("fidelity") for result in results]),
        "rule_precision": _summary([rule.get("precision") for rule in rules]),
        "rule_coverage": _summary([rule.get("coverage") for rule in rules]),
        "rule_complexity": _summary([rule.get("complexity") for rule in rules]),
    }
