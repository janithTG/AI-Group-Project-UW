"""
Simple demo: create random synthetic manufacturing rows and
run the saved final Decision Tree pipeline.

Also compares predictions to the known threshold rule for
Efficiency_Status, so you can see matches, mismatches, and
prediction confidence.

Most runs should look mostly correct (like the real ~99.985%
test accuracy). Rare near-boundary cases are injected only
sometimes so mismatches are not forced every run.

Usage (from project root, with venv active):

    python scripts/demo_inference.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_decision_tree_pipeline.joblib"

# Approximate ranges from the real manufacturing dataset.
FEATURE_RANGES = {
    "Temperature_C": (30.0, 90.0),
    "Vibration_Hz": (0.1, 5.0),
    "Power_Consumption_kW": (1.5, 10.0),
    "Network_Latency_ms": (1.0, 50.0),
    "Packet_Loss_%": (0.0, 5.0),
    "Quality_Control_Defect_Rate_%": (0.0, 10.0),
    "Production_Speed_units_per_hr": (50.0, 500.0),
    "Predictive_Maintenance_Score": (0.0, 1.0),
    "Error_Rate_%": (0.0, 15.0),
}

OPERATION_MODES = ["Active", "Idle", "Maintenance"]
OPERATION_MODE_WEIGHTS = [0.70, 0.20, 0.10]

N_SAMPLES = 10
# Chance to include one difficult near-threshold sample.
BOUNDARY_CASE_PROBABILITY = 0.20

BOUNDARY_CASES = [
    (255.0, 4.9999),  # Expected Medium, tree often predicts Low
    (400.12, 1.40),   # Expected High, tree often predicts Medium
    (382.9, 4.9999),  # Expected Medium, tree often predicts Low
]


def _random_base_features(rng: np.random.Generator) -> dict:
    row = {
        name: float(rng.uniform(low, high))
        for name, (low, high) in FEATURE_RANGES.items()
        if name not in {"Production_Speed_units_per_hr", "Error_Rate_%"}
    }
    row["Operation_Mode"] = str(
        rng.choice(OPERATION_MODES, p=OPERATION_MODE_WEIGHTS)
    )
    return row


def make_synthetic_samples(rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate mostly normal random samples; rarely add one edge case."""
    if rng is None:
        rng = np.random.default_rng()

    rows = []
    for _ in range(N_SAMPLES):
        row = _random_base_features(rng)
        row["Production_Speed_units_per_hr"] = float(
            rng.uniform(*FEATURE_RANGES["Production_Speed_units_per_hr"])
        )
        row["Error_Rate_%"] = float(rng.uniform(*FEATURE_RANGES["Error_Rate_%"]))
        rows.append(row)

    # Only sometimes replace one row with a known hard boundary case.
    if rng.random() < BOUNDARY_CASE_PROBABILITY:
        speed, error = BOUNDARY_CASES[int(rng.integers(0, len(BOUNDARY_CASES)))]
        row = _random_base_features(rng)
        row["Production_Speed_units_per_hr"] = float(
            speed + rng.uniform(-0.05, 0.05)
        )
        row["Error_Rate_%"] = float(error + rng.uniform(-0.00005, 0.00005))
        rows[int(rng.integers(0, N_SAMPLES))] = row

    samples = pd.DataFrame(rows)
    samples = samples.round(
        {
            "Temperature_C": 2,
            "Vibration_Hz": 3,
            "Power_Consumption_kW": 3,
            "Network_Latency_ms": 2,
            "Packet_Loss_%": 3,
            "Quality_Control_Defect_Rate_%": 3,
            "Production_Speed_units_per_hr": 2,
            "Predictive_Maintenance_Score": 3,
            "Error_Rate_%": 3,
        }
    )
    return samples


def expected_efficiency_status(samples: pd.DataFrame) -> pd.Series:
    """
    Ground-truth label from the threshold rule found in the notebooks.

    High:   speed >= 400 and error <= 2
    Medium: speed >= 200 and error <= 5
    Low:    otherwise
    """
    speed = samples["Production_Speed_units_per_hr"]
    error = samples["Error_Rate_%"]

    labels = np.where(
        (speed >= 400) & (error <= 2),
        "High",
        np.where((speed >= 200) & (error <= 5), "Medium", "Low"),
    )
    return pd.Series(labels, index=samples.index, name="Expected")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run notebook 06 first to save the pipeline."
        )

    model = joblib.load(MODEL_PATH)
    samples = make_synthetic_samples()

    predictions = model.predict(samples)
    expected = expected_efficiency_status(samples)
    probabilities = model.predict_proba(samples)
    class_names = list(model.classes_)

    pred_confidence = [
        float(probabilities[i, class_names.index(pred)])
        for i, pred in enumerate(predictions)
    ]

    results = samples[
        [
            "Production_Speed_units_per_hr",
            "Error_Rate_%",
            "Operation_Mode",
        ]
    ].copy()
    results["Expected"] = expected.values
    results["Predicted"] = predictions
    results["Match"] = results["Expected"] == results["Predicted"]
    results["Confidence"] = np.round(pred_confidence, 3)

    n = len(results)
    n_correct = int(results["Match"].sum())
    accuracy = n_correct / n
    error_rate = 1.0 - accuracy

    print("Random synthetic demo predictions:\n")
    print(results.to_string(index=False))
    print("\nBatch summary:")
    print(f"  Samples:     {n}")
    print(f"  Correct:     {n_correct}")
    print(f"  Incorrect:   {n - n_correct}")
    print(f"  Accuracy:    {accuracy:.1%}")
    print(f"  Error rate:  {error_rate:.1%}")
    print(
        "\nNote: Expected uses the known speed/error threshold rule. "
        "Most batches should be fully correct. A hard near-boundary "
        "case is included only ~20% of the time. On the real test set "
        "the model reached about 99.985% accuracy."
    )
    print(f"\nModel used: {MODEL_PATH}")
    print("Run again to see different samples.")


if __name__ == "__main__":
    main()
