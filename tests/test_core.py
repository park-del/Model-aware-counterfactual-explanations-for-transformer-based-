import math

from preprocessing.common import CaseRecord, exact_stratified_split, ratio_counts
from models.tokenizer import ProcessTokenizer
from sage.fitness import compute_fitness, cosine_similarity
from sage.mutation import mutation_distribution
from utils.metrics import gower_distance, levenshtein, mixed_distance, sparsity


def test_exact_split_sizes_and_membership():
    cases = [
        CaseRecord(str(index), {}, {}, ["a", "b"], [None, None], "x" if index % 2 else "y")
        for index in range(100)
    ]
    train, validation, test = exact_stratified_split(cases, (72, 8, 20), seed=42)
    assert tuple(map(len, (train, validation, test))) == (72, 8, 20)
    assert len({case.case_id for case in train + validation + test}) == 100


def test_ratio_counts_sum_to_total():
    assert ratio_counts(1050, (0.72, 0.08, 0.20)) == (756, 84, 210)


def test_fitness_and_cosine():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert compute_fitness(0.8, 0.6, False, alpha=0.5, beta=0.5, gamma=1.0) == 0.7
    assert math.isclose(compute_fitness(0.8, 0.6, True, alpha=0.5, beta=0.5, gamma=1.0), -0.3)


def test_mutation_distribution_is_normalized():
    probabilities = mutation_distribution([1.0, 2.0, 3.0], temperature=0.5)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-12
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_tokenizer_and_paper_distances():
    first = CaseRecord("1", {"amount": 10.0}, {"kind": "a"}, ["x", "y"], [None, None], "no")
    second = CaseRecord("2", {"amount": 20.0}, {"kind": "b"}, ["x", "z"], [None, None], "yes")
    tokenizer = ProcessTokenizer(max_length=8, numeric_bins=2).fit([first, second])
    encoded = tokenizer.encode(first, 2)
    assert len(encoded["input_ids"]) == 8
    assert encoded["attention_mask"].count(1) == 5
    assert levenshtein(first.activities, second.activities) == 1
    assert gower_distance(first, second, {"amount": (10.0, 20.0)}) == 1.0
    assert mixed_distance(first, second, {"amount": (10.0, 20.0)}) == 0.75
    assert sparsity(first, second) == 3
