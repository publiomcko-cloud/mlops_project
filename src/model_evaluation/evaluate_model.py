import json
import logging

import joblib
import mlflow
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from src.mlflow_config import configure_mlflow_tracking


logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model("models/model.keras")


def load_encoder() -> OneHotEncoder:
    return joblib.load("artifacts/[target]_one_hot_encoder.joblib")


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    data_path = "data/processed/test_processed.csv"
    logger.info("Loading test data from %s", data_path)
    data = pd.read_csv(data_path)
    return data.drop("target", axis=1), data["target"]


def load_training_run_id() -> str:
    with open("artifacts/training_run.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    run_id = metadata.get("run_id")
    if not run_id:
        raise RuntimeError("Training metadata does not contain an MLflow run_id.")
    return str(run_id)


def evaluate_model(
    model: tf.keras.Model,
    encoder: OneHotEncoder,
    X: pd.DataFrame,
    y_true: pd.Series,
    run_id: str,
) -> dict:
    configure_mlflow_tracking()
    y_pred_proba = model.predict(X)
    y_pred = encoder.inverse_transform(y_pred_proba).ravel()

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    evaluation = {
        "classification_report": report,
        "confusion_matrix": cm,
    }

    logger.info(
        "Classification Report:\n%s",
        classification_report(y_true, y_pred, zero_division=0),
    )

    with open("metrics/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    # Test metrics are final evaluation evidence only and are never used
    # for model selection.
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("pipeline_stage", "evaluated")
        mlflow.log_metrics(
            {
                "test_accuracy": float(report["accuracy"]),
                "test_precision_weighted": float(
                    report["weighted avg"]["precision"]
                ),
                "test_recall_weighted": float(
                    report["weighted avg"]["recall"]
                ),
                "test_f1_weighted": float(
                    report["weighted avg"]["f1-score"]
                ),
            }
        )

    return evaluation


def main() -> None:
    model = load_model()
    encoder = load_encoder()
    X, y = load_test_data()
    run_id = load_training_run_id()
    evaluate_model(model, encoder, X, y, run_id)
    logger.info("Model evaluation completed for MLflow run %s", run_id)


if __name__ == "__main__":
    main()
