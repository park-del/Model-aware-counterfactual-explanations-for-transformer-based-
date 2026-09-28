"""Shared event-log parsing, splitting, prefixing, and serialization."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass
class Event:
    activity: str
    timestamp: str | None
    attributes: dict[str, Any]


@dataclass
class CaseRecord:
    case_id: str
    numerical: dict[str, float | None]
    categorical: dict[str, str]
    activities: list[str]
    timestamps: list[str | None]
    label: str


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _typed_value(element: ET.Element) -> Any:
    value = element.attrib.get("value")
    kind = _local_name(element.tag)
    if value is None:
        return None
    if kind in {"int", "float"}:
        try:
            return float(value)
        except ValueError:
            return None
    if kind == "boolean":
        return value.lower() == "true"
    return value


def _attributes(element: ET.Element) -> dict[str, Any]:
    return {
        child.attrib["key"]: _typed_value(child)
        for child in element
        if "key" in child.attrib and _local_name(child.tag) != "event"
    }


def iter_xes(path: str | Path) -> Iterator[tuple[dict[str, Any], list[Event]]]:
    """Stream traces from a plain or gzipped XES file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if _local_name(element.tag) != "trace":
                continue
            trace_attributes: dict[str, Any] = {}
            events: list[Event] = []
            for child in element:
                if _local_name(child.tag) == "event":
                    attrs = _attributes(child)
                    events.append(
                        Event(
                            activity=str(attrs.pop("concept:name", "<missing>")),
                            timestamp=attrs.pop("time:timestamp", None),
                            attributes=attrs,
                        )
                    )
                elif "key" in child.attrib:
                    trace_attributes[child.attrib["key"]] = _typed_value(child)
            yield trace_attributes, events
            element.clear()


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    for parser in (
        datetime.fromisoformat,
        lambda text: datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f"),
    ):
        try:
            return parser(normalized)
        except ValueError:
            continue
    return None


def last_non_null(events: Sequence[Event], key: str) -> Any:
    for event in reversed(events):
        value = event.attributes.get(key)
        if value not in (None, ""):
            return value
    return None


def categorical_summary(events: Sequence[Event], key: str) -> dict[str, str]:
    values = [str(e.attributes[key]) for e in events if e.attributes.get(key) not in (None, "")]
    if not values:
        values = ["<missing>"]
    return {
        f"{key}:first": values[0],
        f"{key}:last": values[-1],
        f"{key}:mode": Counter(values).most_common(1)[0][0],
    }


def resolution_hours(timestamps: Sequence[str | None]) -> float:
    parsed = [dt for value in timestamps if (dt := parse_datetime(value)) is not None]
    if len(parsed) < 2:
        return 0.0
    return max(0.0, (max(parsed) - min(parsed)).total_seconds() / 3600.0)


def exact_stratified_split(
    cases: Sequence[CaseRecord], counts: Sequence[int], seed: int
) -> tuple[list[CaseRecord], list[CaseRecord], list[CaseRecord]]:
    """Split into exact train/validation/test sizes while balancing labels."""
    if len(counts) != 3 or sum(counts) != len(cases):
        raise ValueError(f"split counts {tuple(counts)} do not sum to {len(cases)} cases")
    rng = random.Random(seed)
    by_label: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        by_label[case.label].append(case)
    for group in by_label.values():
        rng.shuffle(group)

    targets = list(counts)
    splits: list[list[CaseRecord]] = [[], [], []]
    total = len(cases)
    for label in sorted(by_label):
        group = by_label[label]
        raw = [len(group) * target / total for target in targets]
        allocations = [math.floor(value) for value in raw]
        for index in sorted(range(3), key=lambda i: raw[i] - allocations[i], reverse=True):
            if sum(allocations) == len(group):
                break
            allocations[index] += 1
        cursor = 0
        for split, amount in zip(splits, allocations):
            split.extend(group[cursor : cursor + amount])
            cursor += amount

    for target_index, target in enumerate(targets):
        while len(splits[target_index]) < target:
            donors = [i for i in range(3) if len(splits[i]) > targets[i]]
            donor = max(donors, key=lambda i: len(splits[i]) - targets[i])
            splits[target_index].append(splits[donor].pop())
        while len(splits[target_index]) > target:
            receivers = [i for i in range(3) if len(splits[i]) < targets[i]]
            receiver = min(receivers, key=lambda i: len(splits[i]) - targets[i])
            splits[receiver].append(splits[target_index].pop())
    for split in splits:
        rng.shuffle(split)
    return splits[0], splits[1], splits[2]


def ratio_counts(total: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    train = int(round(total * ratios[0]))
    validation = int(round(total * ratios[1]))
    return train, validation, total - train - validation


def iter_prefix_lengths(case: CaseRecord, minimum: int, maximum: int, step: int) -> Iterator[int]:
    upper = min(maximum, max(1, len(case.activities) - 1))
    if upper < minimum:
        yield upper
    else:
        yield from range(minimum, upper + 1, step)


def save_processed_dataset(
    cases: Sequence[CaseRecord],
    output_dir: str | Path,
    dataset_name: str,
    split_counts: Sequence[int],
    seed: int,
    source_sha256: str,
    paper_feature_counts: dict[str, int],
    assumptions: Sequence[str],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = exact_stratified_split(cases, split_counts, seed)
    splits = {"train": train, "validation": validation, "test": test}
    for name, records in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    metadata = {
        "dataset": dataset_name,
        "source_sha256": source_sha256,
        "num_cases": len(cases),
        "split_counts": {name: len(records) for name, records in splits.items()},
        "labels": sorted({case.label for case in cases}),
        "activities": sorted({activity for case in cases for activity in case.activities}),
        "numerical_features": sorted({key for case in cases for key in case.numerical}),
        "categorical_features": sorted({key for case in cases for key in case.categorical}),
        "paper_feature_counts": paper_feature_counts,
        "assumptions": list(assumptions),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def read_jsonl(path: str | Path) -> list[CaseRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        return [CaseRecord(**json.loads(line)) for line in handle]
