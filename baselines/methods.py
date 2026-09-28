"""Paper-compatible adapters for the four counterfactual baselines.

The paper does not publish source code or every baseline hyperparameter. These
implementations follow Table 7 and share SAGE's tokenizer, predictor, domains,
and feasibility repair so comparisons remain controlled.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import asdict

import numpy as np
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph

from sage.counterfactual import SAGEExplainer
from sage.crossover import crossover
from sage.fitness import semantic_similarity
from utils.metrics import mixed_distance, numeric_ranges, sparsity


class BaselineSearch:
    def __init__(self, model, tokenizer, reference_cases, config, device=None):
        self.adapter = SAGEExplainer(model, tokenizer, reference_cases, config, device)
        self.tokenizer = tokenizer
        self.reference_cases = list(reference_cases)
        self.config = config
        self.rng = random.Random(int(config.get("seed", 42)))
        self.ranges = numeric_ranges(self.reference_cases)

    def _context(self, original_case, prefix_length, target_class=None):
        original = self.adapter._prefix(original_case, prefix_length)
        original_output = self.adapter._model_outputs([original])
        probabilities = original_output["probabilities"][0]
        original_id = int(probabilities.argmax())
        if target_class is None:
            target_id = int(np.argsort(probabilities)[-2])
        elif isinstance(target_class, str):
            target_id = self.tokenizer.label_to_id[target_class]
        else:
            target_id = int(target_class)
        pool, pool_output = self.adapter._reference_at_length(prefix_length)
        return original, original_output, original_id, target_id, pool, pool_output

    def _uniform_mutate(self, candidate):
        numeric = [("numerical", key) for key in self.tokenizer.numerical_features]
        categorical = [("categorical", key) for key in self.tokenizer.categorical_features]
        dynamic = [("activity", index) for index in range(len(candidate.activities))]
        block, key = self.rng.choice(numeric + categorical + dynamic)
        if block == "activity":
            values = [
                value
                for value in self.adapter.domains["activities"].get(key, self.adapter.domains["all_activities"])
                if value != candidate.activities[key]
            ]
            if values:
                candidate.activities[key] = self.rng.choice(values)
        else:
            current = getattr(candidate, block).get(key)
            values = [value for value in self.adapter.domains[block][key] if value != current]
            if values:
                getattr(candidate, block)[key] = self.rng.choice(values)
        return self.adapter._repair(candidate)

    def _pack(self, method, original, original_id, target_id, candidate=None):
        payload = {
            "method": method,
            "original": asdict(original),
            "original_prediction": self.tokenizer.id_to_label[original_id],
            "target_class": self.tokenizer.id_to_label[target_id],
            "prefix_length": len(original.activities),
            "counterfactuals": [],
            "validity": candidate is not None,
        }
        if candidate is None:
            return payload
        original_output = self.adapter._model_outputs([original])
        candidate_output = self.adapter._model_outputs([candidate])
        similarity = semantic_similarity(
            candidate_output["static"][0],
            candidate_output["dynamic"][0],
            original_output["static"][0],
            original_output["dynamic"][0],
            float(self.config.get("dynamic_similarity_weight", 0.7)),
        )
        payload["counterfactuals"] = [
            {
                "case": asdict(candidate),
                "target_probability": float(candidate_output["probabilities"][0, target_id]),
                "semantic_similarity": float(similarity),
                "fitness": None,
                "changed_features": sparsity(original, candidate),
            }
        ]
        return payload

    def face(self, original_case, prefix_length, target_class=None, neighbors=10):
        original, original_output, original_id, target_id, pool, pool_output = self._context(
            original_case, prefix_length, target_class
        )
        nodes = [original] + pool
        features = np.concatenate(
            [
                np.concatenate([original_output["static"], original_output["dynamic"]], axis=1),
                np.concatenate([pool_output["static"], pool_output["dynamic"]], axis=1),
            ],
            axis=0,
        )
        graph = kneighbors_graph(
            features,
            n_neighbors=min(neighbors, len(nodes) - 1),
            mode="distance",
            include_self=False,
        )
        graph = graph.maximum(graph.T)
        distances = dijkstra(graph, directed=False, indices=0)
        predictions = np.concatenate(
            [[original_id], pool_output["probabilities"].argmax(axis=1)]
        )
        targets = [index for index in range(1, len(nodes)) if predictions[index] == target_id and np.isfinite(distances[index])]
        candidate = nodes[min(targets, key=lambda index: distances[index])] if targets else None
        return self._pack("FACE", original, original_id, target_id, candidate)

    def growing_spheres(
        self,
        original_case,
        prefix_length,
        target_class=None,
        initial_radius=0.10,
        radius_step=0.10,
        samples_per_layer=1000,
    ):
        original, _, original_id, target_id, _, _ = self._context(original_case, prefix_length, target_class)
        total_features = len(self.tokenizer.numerical_features) + len(self.tokenizer.categorical_features) + len(original.activities)
        radius = initial_radius
        while radius <= 1.0 + 1e-12:
            mutations = max(1, math.ceil(radius * total_features))
            candidates = []
            for _ in range(samples_per_layer):
                candidate = deepcopy(original)
                for _ in range(mutations):
                    self._uniform_mutate(candidate)
                candidates.append(candidate)
            outputs = self.adapter._model_outputs(candidates)
            valid = [index for index, value in enumerate(outputs["probabilities"].argmax(axis=1)) if value == target_id]
            if valid:
                best = min(valid, key=lambda index: mixed_distance(original, candidates[index], self.ranges))
                return self._pack("Growing Spheres", original, original_id, target_id, candidates[best])
            radius += radius_step
        return self._pack("Growing Spheres", original, original_id, target_id)

    def _genetic(self, method, original_case, prefix_length, target_class, population_size, generations, pc, pm, proximity_weight):
        original, _, original_id, target_id, pool, _ = self._context(original_case, prefix_length, target_class)
        population = [deepcopy(self.rng.choice(pool)) for _ in range(population_size)]
        for _ in range(generations):
            outputs = self.adapter._model_outputs(population)
            fitness = np.asarray(
                [
                    probability[target_id]
                    + proximity_weight * (1.0 - mixed_distance(original, candidate, self.ranges))
                    for candidate, probability in zip(population, outputs["probabilities"])
                ]
            )
            elite = [deepcopy(population[index]) for index in np.argsort(-fitness)[: max(1, population_size // 20)]]
            next_population = elite
            while len(next_population) < population_size:
                tournament = self.rng.sample(range(population_size), min(3, population_size))
                first = deepcopy(population[max(tournament, key=lambda index: fitness[index])])
                tournament = self.rng.sample(range(population_size), min(3, population_size))
                second = deepcopy(population[max(tournament, key=lambda index: fitness[index])])
                if self.rng.random() < pc:
                    first, second = crossover(first, second, self.rng)
                for child in (first, second):
                    if self.rng.random() < pm:
                        self._uniform_mutate(child)
                    next_population.append(self.adapter._repair(child))
                    if len(next_population) >= population_size:
                        break
            population = next_population
        outputs = self.adapter._model_outputs(population)
        valid = [index for index, value in enumerate(outputs["probabilities"].argmax(axis=1)) if value == target_id]
        candidate = None
        if valid:
            candidate = max(
                (population[index] for index in valid),
                key=lambda item: 1.0 - mixed_distance(original, item, self.ranges),
            )
        return self._pack(method, original, original_id, target_id, candidate)

    def dice(self, original_case, prefix_length, target_class=None):
        return self._genetic("DiCE", original_case, prefix_length, target_class, 1000, 10, 0.7, 0.2, 0.25)

    def loreley(self, original_case, prefix_length, target_class=None):
        return self._genetic("LORELEY", original_case, prefix_length, target_class, 1000, 10, 0.8, 0.2, 0.75)

