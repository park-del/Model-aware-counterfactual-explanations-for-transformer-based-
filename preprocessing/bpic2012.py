"""BPIC 2012 preprocessing and three-class loan outcome construction."""

from preprocessing.common import (
    CaseRecord,
    categorical_summary,
    file_sha256,
    iter_xes,
    parse_datetime,
    ratio_counts,
    save_processed_dataset,
)


OUTCOMES = {"A_APPROVED": "approved", "A_DECLINED": "declined", "A_CANCELLED": "cancelled"}


def preprocess_bpic2012(input_path, output_dir, config):
    cases = []
    skipped = 0
    for trace, events in iter_xes(input_path):
        activities = [event.activity for event in events]
        outcome = next((OUTCOMES[name] for name in OUTCOMES if name in activities), None)
        if outcome is None:
            skipped += 1
            continue
        registered = parse_datetime(str(trace.get("REG_DATE", "")))
        numerical = {
            "AMOUNT_REQ": float(trace.get("AMOUNT_REQ")) if trace.get("AMOUNT_REQ") is not None else None,
            "registration_year": float(registered.year) if registered else None,
            "registration_month": float(registered.month) if registered else None,
            "registration_weekday": float(registered.weekday()) if registered else None,
            "registration_hour": float(registered.hour) if registered else None,
            "registration_minute": float(registered.minute) if registered else None,
        }
        categorical = {
            "registration_month": str(registered.month) if registered else "<missing>",
            "registration_weekday": str(registered.weekday()) if registered else "<missing>",
            "registration_hour_bin": str(registered.hour // 4) if registered else "<missing>",
        }
        for key in ("org:resource", "lifecycle:transition"):
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
    counts = tuple(config.get("split_counts", ()))
    if sum(counts) != len(cases):
        counts = ratio_counts(len(cases), config.get("split_ratios", (0.72, 0.08, 0.20)))
    return save_processed_dataset(
        cases,
        output_dir,
        "bpic2012",
        counts,
        int(config.get("seed", 42)),
        file_sha256(input_path),
        {"static_numerical": 6, "static_categorical": 24, "dynamic_activities": 24},
        [
            f"Excluded {skipped} traces without A_APPROVED, A_DECLINED, or A_CANCELLED; the paper does not specify their label.",
            "The paper reports 24 static categorical features but does not publish their field mapping; only non-target raw summaries are emitted.",
        ],
    )
