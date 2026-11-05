import numpy as np
import pandas as pd
import yaml

from src.data_preprocessing.preprocess_data import preprocess_data
from src.feature_engineering.engineer_features import engineer_features


def test_imputer_is_fit_on_training_data_only():
    train = pd.DataFrame(
        {
            "feature": [1.0, 3.0, np.nan],
            "target": [0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "feature": [100.0, np.nan],
            "target": [1, 0],
        }
    )

    _, processed_test, imputer = preprocess_data(train, test)

    assert imputer.statistics_[0] == 2.0
    assert processed_test.loc[1, "feature"] == 2.0


def test_scaler_is_fit_on_training_data_only():
    train = pd.DataFrame(
        {
            "feature": [1.0, 3.0],
            "target": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "feature": [100.0],
            "target": [1],
        }
    )

    _, transformed_test, scaler = engineer_features(train, test)

    assert scaler.mean_[0] == 2.0
    assert transformed_test.loc[0, "feature"] > 10


def test_dvc_pipeline_tracks_training_run_metadata():
    with open("dvc.yaml", "r", encoding="utf-8") as f:
        pipeline = yaml.safe_load(f)

    train_outs = pipeline["stages"]["train"]["outs"]
    evaluate_deps = pipeline["stages"]["evaluate"]["deps"]

    assert "artifacts/training_run.json" in train_outs
    assert "artifacts/training_run.json" in evaluate_deps
