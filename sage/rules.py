"""Local surrogate decision-tree rules and reliability metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree


def _leaf_paths(tree):
    paths = []

    def visit(node, conditions):
        if tree.children_left[node] == _tree.TREE_LEAF:
            paths.append((node, conditions))
            return
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        visit(tree.children_left[node], conditions + [(feature, "<=", threshold)])
        visit(tree.children_right[node], conditions + [(feature, ">", threshold)])

    visit(0, [])
    return paths


def _satisfies(row, conditions):
    return all((row[index] <= threshold) if operator == "<=" else (row[index] > threshold) for index, operator, threshold in conditions)


def extract_counterfactual_rules(features, black_box_labels, original_features, target_class, max_depth=5, seed=42):
    features = np.asarray(features)
    labels = np.asarray(black_box_labels)
    if len(np.unique(labels)) < 2:
        return {"fidelity": 1.0, "rules": []}
    surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    surrogate.fit(features, labels)
    predictions = surrogate.predict(features)
    rules = []
    for leaf, conditions in _leaf_paths(surrogate.tree_):
        leaf_class = int(surrogate.classes_[np.argmax(surrogate.tree_.value[leaf][0])])
        if leaf_class != int(target_class) or _satisfies(original_features, conditions):
            continue
        covered = np.asarray([_satisfies(row, conditions) for row in features])
        if not covered.any():
            continue
        rules.append(
            {
                "conditions": [
                    {"token_position": index, "operator": operator, "threshold_token_id": threshold}
                    for index, operator, threshold in conditions
                ],
                "coverage": float(covered.mean()),
                "precision": float((labels[covered] == target_class).mean()),
                "complexity": len(conditions),
            }
        )
    return {"fidelity": float(accuracy_score(labels, predictions)), "rules": rules}

