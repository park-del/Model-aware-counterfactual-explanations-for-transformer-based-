"""Sepsis preprocessing and binary return-to-ER outcome construction."""

from preprocessing.common import CaseRecord, file_sha256, iter_xes, ratio_counts, save_processed_dataset


NUMERICAL_KEYS = ("Age", "Leucocytes", "CRP", "LacticAcid")
EXCLUDED = {"concept:name", "time:timestamp", "lifecycle:transition", *NUMERICAL_KEYS}


def preprocess_sepsis(input_path, output_dir, config):
    cases = []
    for trace, events in iter_xes(input_path):
        activities = [event.activity for event in events]
        first = events[0].attributes if events else {}
        numerical = {}
        for key in NUMERICAL_KEYS:
            value = next((event.attributes.get(key) for event in events if event.attributes.get(key) is not None), None)
            numerical[key] = float(value) if value is not None else None
        categorical = {
            key: str(value)
            for key, value in first.items()
            if key not in EXCLUDED and value not in (None, "")
        }
        cases.append(
            CaseRecord(
                case_id=str(trace.get("concept:name")),
                numerical=numerical,
                categorical=categorical,
                activities=activities,
                timestamps=[event.timestamp for event in events],
                label="return_er" if "Return ER" in activities else "no_return_er",
            )
        )
    counts = tuple(config.get("split_counts", ()))
    if sum(counts) != len(cases):
        counts = ratio_counts(len(cases), config.get("split_ratios", (0.72, 0.08, 0.20)))
    return save_processed_dataset(
        cases,
        output_dir,
        "sepsis",
        counts,
        int(config.get("seed", 42)),
        file_sha256(input_path),
        {"static_numerical": 4, "static_categorical": 32, "dynamic_activities": 16},
        [
            "The binary outcome is presence versus absence of Return ER in the completed trace.",
            "The paper calls this a discharge-outcome task but does not publish the exact label mapping.",
        ],
    )
