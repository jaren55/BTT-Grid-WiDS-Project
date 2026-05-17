from __future__ import annotations

import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from src.features import make_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train_processed.csv"
DEFAULT_BATCH_PATH = PROJECT_ROOT / "data" / "test_processed.csv"
HORIZONS = np.array([12, 24, 48, 72], dtype=float)
MANUAL_INPUT_COLUMNS = [
    "dist_min_ci_0_5h",
    "alignment_abs",
    "closing_speed_m_per_h",
    "dt_first_last_0_5h",
    "num_perimeters_0_5h",
    "spread_bearing_cos",
]


def _clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1e-4, 1 - 1e-4)


def load_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    train_df = pd.read_csv(TRAIN_PATH)
    x_train = make_features(train_df)
    y_train = np.array(
        list(zip(train_df["event"].astype(bool), train_df["time_to_hit_hours"].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )
    return x_train, y_train


@lru_cache(maxsize=1)
def load_models() -> tuple[
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
    CoxPHSurvivalAnalysis,
    np.ndarray,
]:
    x_train, y_train = load_training_data()

    rsf = RandomSurvivalForest(
        n_estimators=2000,
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=3,
        max_features="log2",
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(x_train, y_train)

    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    gbsa.fit(x_train, y_train)

    coxph = CoxPHSurvivalAnalysis()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        coxph.fit(x_train, pd.DataFrame(y_train).to_records(index=False))

    return rsf, gbsa, coxph, rsf.unique_times_


def _cox_survival_matrix(
    coxph: CoxPHSurvivalAnalysis, x_input: pd.DataFrame, unique_times: np.ndarray
) -> np.ndarray:
    baseline_survival = coxph.baseline_survival_
    risk_scores = coxph.predict(x_input)

    cox_survival = np.zeros((x_input.shape[0], len(unique_times)))
    for i, horizon in enumerate(unique_times):
        cox_survival[:, i] = baseline_survival(horizon) ** np.exp(risk_scores)

    return cox_survival


def predict_probabilities(input_df: pd.DataFrame) -> pd.DataFrame:
    rsf, gbsa, coxph, unique_times = load_models()
    x_input = make_features(input_df)

    surv_rsf = rsf.predict_survival_function(x_input, return_array=True)
    surv_gbsa = gbsa.predict_survival_function(x_input, return_array=True)
    surv_cox = _cox_survival_matrix(coxph, x_input, unique_times)

    ensemble_survival = 0.7 * surv_rsf + 0.15 * surv_gbsa + 0.05 * surv_cox

    time_indices = []
    for horizon in HORIZONS:
        idx = np.searchsorted(unique_times, horizon, side="right") - 1
        idx = int(np.clip(idx, 0, len(unique_times) - 1))
        time_indices.append(idx)

    predictions = pd.DataFrame(index=input_df.index.copy())
    for horizon, idx in zip(HORIZONS.astype(int), time_indices):
        predictions[f"prob_{horizon}h"] = _clip_probs(1 - ensemble_survival[:, idx])

    return predictions.reset_index(drop=True)


def predict_manual(inputs: dict[str, float]) -> pd.DataFrame:
    manual_df = pd.DataFrame([{column: float(inputs[column]) for column in MANUAL_INPUT_COLUMNS}])
    return predict_probabilities(manual_df)


def score_batch_file(file_path: str | Path) -> pd.DataFrame:
    batch_df = pd.read_csv(file_path)
    predictions = predict_probabilities(batch_df)

    if "event_id" in batch_df.columns:
        return pd.concat([batch_df[["event_id"]].reset_index(drop=True), predictions], axis=1)

    return predictions


def export_predictions(results_df: pd.DataFrame) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_predictions.csv", prefix="wildfire_", delete=False
    ) as handle:
        results_df.to_csv(handle.name, index=False)
        return handle.name
