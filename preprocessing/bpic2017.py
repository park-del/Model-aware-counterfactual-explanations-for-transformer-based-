"""BPIC 2017 preprocessing and three-class outcome construction."""

from preprocessing.common import (
    CaseRecord,
    categorical_summary,
    file_sha256,
    iter_xes,
    last_non_null,
    ratio_counts,
    save_processed_dataset,
)


OUTCOMES = {"A_Pending": "pending", "A_Denied": "denied", "A_Cancelled": "cancelled"}


def preprocess_bpic2017(input_path, output_dir, config):
    cases = []
    skipped = 0
    for trace, events in iter_xes(input_path):
        activities = [event.activity for event in events]
        outcome = next((OUTCOMES[name] for name in OUTCOMES if name in activities), None)
        if outcome is None:
            skipped += 1
            continue
        numerical = {
            "RequestedAmount": float(trace.get("RequestedAmount")) if trace.get("RequestedAmount") is not None else None,
            **{
                key: float(value) if (value := last_non_null(events, key)) is not None else None
                for key in ("FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost", "CreditScore", "OfferedAmount")
            },
        }
        categorical = {
            "LoanGoal": str(trace.get("LoanGoal", "<missing>")),
            "ApplicationType": str(trace.get("ApplicationType", "<missing>")),
        }
        for key in ("Action", "EventOrigin", "org:resource", "lifecycle:transition", "Accepted", "Selected"):
            categorical.update(categorical_summary(events, key))
        cases.append(
            CaseRecord(
                case_id=str(trace.get("concept:name")),
                numerical=numerical,
                categorical=categorical,
                activities=activities,
                timestamps=[event.timestamp for event in events],
                label=outcome,
            )
        )
    ratios = config.get("split_ratios", (0.72, 0.08, 0.20))
    counts = tuple(config.get("split_counts", ()))
    if sum(counts) != len(cases):
        counts = ratio_counts(len(cases), ratios)
    return save_processed_dataset(
        cases,
        output_dir,
        "bpic2017",
        counts,
        int(config.get("seed", 42)),
        file_sha256(input_path),
        {"static_numerical": 6, "static_categorical": 20, "dynamic_activities": 26},
        [
            f"Excluded {skipped} traces without A_Pending, A_Denied, or A_Cancelled; the paper does not specify their label.",
            "Offer attributes use their last observed non-null value.",
            "Categorical event attributes are summarized by first, last, and modal values.",
        ],
    )
