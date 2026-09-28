"""End-to-end Semantic Attention-Guided counterfactual search."""

from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

from sage.crossover import crossover
from sage.fitness import compute_fitness, semantic_similarity
from sage.mutation import mutate
from sage.rules import extract_counterfactual_rules
from sage.semantic_initialization import initialize_population


class SAGEExplainer:
    def __init__(self, model, tokenizer, reference_cases, config, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_cases = list(reference_cases)
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.rng = random.Random(int(config.get("seed", 42)))
        self._reference_cache = {}
        self.domains = self._build_domains()
        self.direct_follow = {
            pair
            for case in self.reference_cases
            for pair in zip(case.activities, case.activities[1:])
        }

    def _build_domains(self):
        numerical, categorical = defaultdict(set), defaultdict(set)
        activities, all_activities = defaultdict(set), set()
        for case in self.reference_cases:
            for key, value in case.numerical.items():
                numerical[key].add(value)
            for key, value in case.categorical.items():
                categorical[key].add(value)
            for position, activity in enumerate(case.activities):
                activities[position].add(activity)
                all_activities.add(activity)
        neighborhood_features = np.asarray(
            [self.tokenizer.encode(candidate, len(candidate.activities))["input_ids"] for candidate in population]
        )
        original_features = np.asarray(self.tokenizer.encode(original, len(original.activities))["input_ids"])
        surrogate = extract_counterfactual_rules(
            neighborhood_features,
            predictions,
            original_features,
            target_id,
            int(self.config.get("surrogate_max_depth", 5)),
            int(self.config.get("seed", 42)),
        )
        return {
            "numerical": {key: sorted(values, key=lambda value: (value is None, str(value))) for key, values in numerical.items()},
            "categorical": {key: sorted(values) for key, values in categorical.items()},
            "activities": {key: sorted(values) for key, values in activities.items()},
            "all_activities": sorted(all_activities),
        }

    @staticmethod
    def _same(a, b):
        return a.numerical == b.numerical and a.categorical == b.categorical and a.activities == b.activities

    @staticmethod
    def _changes(a, b):
        static = sum(a.numerical.get(key) != b.numerical.get(key) for key in set(a.numerical) | set(b.numerical))
        static += sum(a.categorical.get(key) != b.categorical.get(key) for key in set(a.categorical) | set(b.categorical))
        dynamic = sum(left != right for left, right in zip(a.activities, b.activities))
        return static + dynamic

    def _prefix(self, case, length):
        candidate = deepcopy(case)
        candidate.activities = candidate.activities[:length]
        candidate.timestamps = candidate.timestamps[:length]
        return candidate

    def _model_outputs(self, cases, return_attention=False, batch_size=256):
        probabilities, static_vectors, dynamic_vectors, attention_scores = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(cases), batch_size):
                chunk = cases[start : start + batch_size]
                encoded = [self.tokenizer.encode(case, len(case.activities)) for case in chunk]
                batch = {
                    key: torch.tensor([item[key] for item in encoded], dtype=torch.long, device=self.device)
                    for key in ("input_ids", "type_ids", "position_ids", "attention_mask")
                }
                output = self.model(**batch, return_attention=True)
                hidden = output["hidden_states"]
                for type_id, destination in ((1, static_vectors), (2, dynamic_vectors)):
                    mask = (batch["type_ids"] == type_id) & batch["attention_mask"].bool()
                    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                    destination.extend(pooled.cpu().numpy())
                probabilities.extend(output["logits"].softmax(dim=-1).cpu().numpy())
                if return_attention:
                    attention_scores.extend(output["aggregated_attention"].sum(dim=1).cpu().numpy())
        result = {
            "probabilities": np.asarray(probabilities),
            "static": np.asarray(static_vectors),
            "dynamic": np.asarray(dynamic_vectors),
        }
        if return_attention:
            result["attention"] = np.asarray(attention_scores)
        return result

    def _repair(self, candidate):
        for position in range(1, len(candidate.activities)):
            left, current = candidate.activities[position - 1], candidate.activities[position]
            if (left, current) in self.direct_follow:
                continue
            following = candidate.activities[position + 1] if position + 1 < len(candidate.activities) else None
            options = [
                value
                for value in self.domains["activities"].get(position, self.domains["all_activities"])
                if (left, value) in self.direct_follow
                and (following is None or (value, following) in self.direct_follow)
            ]
            if options:
                candidate.activities[position] = self.rng.choice(options)
        return candidate

    def _tournament(self, population, fitness):
        size = min(int(self.config.get("tournament_size", 3)), len(population))
        indices = self.rng.sample(range(len(population)), size)
        return deepcopy(population[max(indices, key=lambda index: fitness[index])])

    def predict(self, case, prefix_length):
        output = self._model_outputs([self._prefix(case, prefix_length)])
        probabilities = output["probabilities"][0]
        prediction = int(probabilities.argmax())
        return self.tokenizer.id_to_label[prediction], probabilities

    def _reference_at_length(self, prefix_length):
        if prefix_length not in self._reference_cache:
            pool = [self._prefix(case, prefix_length) for case in self.reference_cases if len(case.activities) >= prefix_length]
            self._reference_cache[prefix_length] = (pool, self._model_outputs(pool))
        return self._reference_cache[prefix_length]

    def explain(self, original_case, prefix_length, target_class=None, top_k=5):
        original = self._prefix(original_case, prefix_length)
        original_output = self._model_outputs([original], return_attention=True)
        original_probabilities = original_output["probabilities"][0]
        original_class = int(original_probabilities.argmax())
        if target_class is None:
            target_id = int(np.argsort(original_probabilities)[-2])
        elif isinstance(target_class, str):
            target_id = self.tokenizer.label_to_id[target_class]
        else:
            target_id = int(target_class)
        if target_id == original_class:
            raise ValueError("target class must differ from the original prediction")

        pool, pool_output = self._reference_at_length(prefix_length)
        population_size = int(self.config.get("population_size", 1000))
        if self.config.get("initialization", "semantic") == "random":
            population = [deepcopy(value) for value in self.rng.sample(pool, min(population_size, len(pool)))]
        else:
            population, _ = initialize_population(
                pool,
                pool_output["static"],
                pool_output["dynamic"],
                original_output["static"][0],
                original_output["dynamic"][0],
                min(population_size, len(pool)),
                float(self.config.get("dynamic_similarity_weight", 0.7)),
            )
        if not population:
            raise ValueError(f"no reference trace has prefix length {prefix_length}")
        while len(population) < population_size:
            population.append(deepcopy(self.rng.choice(population)))

        for _ in range(int(self.config.get("generations", 10))):
            output = self._model_outputs(population, return_attention=True)
            similarities = np.asarray(
                [
                    semantic_similarity(
                        static,
                        dynamic,
                        original_output["static"][0],
                        original_output["dynamic"][0],
                        float(self.config.get("dynamic_similarity_weight", 0.7)),
                    )
                    for static, dynamic in zip(output["static"], output["dynamic"])
                ]
            )
            fitness = np.asarray(
                [
                    compute_fitness(
                        0.0 if self.config.get("fitness_mode") == "distance_only" else probability[target_id],
                        similarity,
                        self._same(candidate, original),
                        0.0 if self.config.get("fitness_mode") == "distance_only" else float(self.config.get("alpha", 0.5)),
                        1.0 if self.config.get("fitness_mode") == "distance_only" else float(self.config.get("beta", 0.5)),
                        float(self.config.get("gamma", 1.0)),
                    )
                    for candidate, probability, similarity in zip(population, output["probabilities"], similarities)
                ]
            )
            order = np.argsort(-fitness)
            elite_count = max(1, int(np.ceil(float(self.config.get("elitism_rate", 0.05)) * population_size)))
            next_population = [deepcopy(population[index]) for index in order[:elite_count]]
            while len(next_population) < population_size:
                first = self._tournament(population, fitness)
                second = self._tournament(population, fitness)
                if self.rng.random() < float(self.config.get("crossover_probability", 0.7)):
                    first, second = crossover(first, second, self.rng)
                for child in (first, second):
                    if self.rng.random() < float(self.config.get("mutation_probability", 0.2)):
                        child_output = self._model_outputs([child], return_attention=True)
                        attention_scores = child_output["attention"][0]
                        if self.config.get("mutation_strategy", "attention") == "random":
                            attention_scores = np.ones_like(attention_scores)
                        mutate(
                            child,
                            attention_scores,
                            self.domains,
                            self.tokenizer,
                            self.rng,
                            float(self.config.get("temperature", 0.5)),
                        )
                    if self.config.get("enforce_feasibility", True):
                        child = self._repair(child)
                    next_population.append(child)
                    if len(next_population) >= population_size:
                        break
            population = next_population

        output = self._model_outputs(population)
        similarities = np.asarray(
            [
                semantic_similarity(
                    static,
                    dynamic,
                    original_output["static"][0],
                    original_output["dynamic"][0],
                    float(self.config.get("dynamic_similarity_weight", 0.7)),
                )
                for static, dynamic in zip(output["static"], output["dynamic"])
            ]
        )
        fitness = np.asarray(
            [
                compute_fitness(
                    0.0 if self.config.get("fitness_mode") == "distance_only" else probability[target_id],
                    similarity,
                    self._same(candidate, original),
                    0.0 if self.config.get("fitness_mode") == "distance_only" else float(self.config.get("alpha", 0.5)),
                    1.0 if self.config.get("fitness_mode") == "distance_only" else float(self.config.get("beta", 0.5)),
                    float(self.config.get("gamma", 1.0))
                )
                for candidate, probability, similarity in zip(population, output["probabilities"], similarities)
            ]
        )
        predictions = output["probabilities"].argmax(axis=1)
        order = np.argsort(-fitness)
        selected, seen = [], set()
        target_pool = [index for index, case in enumerate(pool) if self.tokenizer.label_to_id.get(case.label) == target_id]
        for index in order:
            if predictions[index] != target_id:
                continue
            candidate = population[index]
            signature = (tuple(sorted(candidate.numerical.items())), tuple(sorted(candidate.categorical.items())), tuple(candidate.activities))
            if signature in seen:
                continue
            seen.add(signature)
            if target_pool:
                plausibility = max(
                    semantic_similarity(
                        output["static"][index],
                        output["dynamic"][index],
                        pool_output["static"][reference_index],
                        pool_output["dynamic"][reference_index],
                        float(self.config.get("dynamic_similarity_weight", 0.7)),
                    )
                    for reference_index in target_pool
                )
            else:
                plausibility = float("nan")
            selected.append(
                {
                    "case": asdict(candidate),
                    "target_probability": float(output["probabilities"][index, target_id]),
                    "semantic_similarity": float(similarities[index]),
                    "fitness": float(fitness[index]),
                    "changed_features": self._changes(candidate, original),
                    "plausibility": float(plausibility),
                }
            )
            if len(selected) >= top_k:
                break
        neighborhood_features = np.asarray(
            [self.tokenizer.encode(candidate, len(candidate.activities))["input_ids"] for candidate in population]
        )
        original_features = np.asarray(self.tokenizer.encode(original, len(original.activities))["input_ids"])
        surrogate = extract_counterfactual_rules(
            neighborhood_features,
            predictions,
            original_features,
            target_id,
            int(self.config.get("surrogate_max_depth", 5)),
            int(self.config.get("seed", 42)),
        )
        return {
            "original": asdict(original),
            "original_prediction": self.tokenizer.id_to_label[original_class],
            "target_class": self.tokenizer.id_to_label[target_id],
            "prefix_length": prefix_length,
            "counterfactuals": selected,
            "validity": bool(selected),
            "surrogate": surrogate,
        }
