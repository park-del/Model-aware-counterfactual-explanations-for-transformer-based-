"""Helpdesk CSV preprocessing and configurable timely/delayed labels."""

import csv
import statistics
from collections import defaultdict

from preprocessing.common import (
    CaseRecord,
    file_sha256,
    parse_datetime,
    ratio_counts,
    resolution_hours,
    save_processed_dataset,
)


CATEGORICAL_COLUMNS = (
    "seriousness",
    "customer",
    "product",
    "responsible_section",
    "seriousness_2",
    "service_level",
    "service_type",
    "support_section",
    "workgroup",
)


def preprocess_helpdesk(input_path, output_dir, config):
    grouped = defaultdict(list)
    with open(input_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            grouped[row["Case ID"]].append(row)

    prepared = []
    for case_id, rows in grouped.items():
        rows.sort(key=lambda row: parse_datetime(row["Complete Timestamp"]) or parse_datetime("1970-01-01"))
        timestamps = [row["Complete Timestamp"] for row in rows]
        start = parse_datetime(timestamps[0])
        duration = resolution_hours(timestamps)
        prepared.append((case_id, rows, timestamps, start, duration))
    threshold = config.get("timely_threshold_hours")
    if threshold is None:
        threshold = statistics.median(item[4] for item in prepared)

    cases = []
    for case_id, rows, timestamps, start, duration in prepared:
        first = rows[0]
        numerical = {
            "start_year": float(start.year) if start else None,
            "start_month": float(start.month) if start else None,
            "start_weekday": float(start.weekday()) if start else None,
            "start_hour": float(start.hour) if start else None,
            "variant_index": float(first["Variant index"]) if first.get("Variant index") else None,
        }
        cases.append(
            CaseRecord(
                case_id=case_id,
                numerical=numerical,
                categorical={key: first.get(key, "<missing>") for key in CATEGORICAL_COLUMNS},
                activities=[row["Activity"] for row in rows],
                timestamps=timestamps,
                label="timely" if duration <= float(threshold) else "delayed",
            )
        )
    counts = tuple(config.get("split_counts", ()))
    if sum(counts) != len(cases):
        counts = ratio_counts(len(cases), config.get("split_ratios", (0.72, 0.08, 0.20)))
    return save_processed_dataset(
        cases,
        output_dir,
        "helpdesk",
        counts,
        int(config.get("seed", 42)),
        file_sha256(input_path),
        {"static_numerical": 5, "static_categorical": 25, "dynamic_activities": 14},
        [
            f"Timely/delayed threshold is {float(threshold):.6f} hours.",
            "The paper defines the label using total resolution time but does not publish the threshold; the default is the dataset median.",
            "The paper reports 25 static categorical features but does not publish their field mapping; the nine raw case attributes are retained.",
        ],
    )
