import json
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger("src.model_training.train_model")


def load_data() -> pd.DataFrame:
    train_path = "data/processed/train_processed.csv"
    logger.info("Loading feature data from %s", train_path)
    return pd.read_csv(train_path)


def load_params() -> dict[str, float | int]:
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params["train"]


def prepare_data(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, OneHotEncoder]:
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
    return X_train, y_train_encoded, encoder


def create_model(
    input_shape: int,
    num_classes: int,
    params: dict[str, int | float],
) -> tf.keras.Model:
    model = Sequential(
        [
            tf.keras.Input(shape=(input_shape,)),
            Dense(params["hidden_layer_1_neurons"], activation="relu"),
            Dropout(params["dropout_rate"]),
            Dense(params["hidden_layer_2_neurons"], activation="relu"),
            Dropout(params["dropout_rate"]),
            Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_artifacts(
    model: tf.keras.Model,
    encoder: OneHotEncoder,
) -> None:
    model_path = os.path.join("models", "model.keras")
    encoder_path = os.path.join(
        "artifacts",
        "[target]_one_hot_encoder.joblib",
    )

    logger.info("Saving model to %s", model_path)
    model.save(model_path)
    logger.info("Saving encoder to %s", encoder_path)
    joblib.dump(encoder, encoder_path)


def _mlflow_run_arguments(experiment_id: str) -> dict:
    dvc_exp_name = os.getenv("DVC_EXP_NAME")
    if not dvc_exp_name:
        return {}

    parent_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.dvc_exp = 'True'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if parent_runs.empty:
        with mlflow.start_run(run_name=f"{dvc_exp_name}-parent") as parent_run:
            mlflow.set_tag("dvc_exp", True)
            parent_run_id = parent_run.info.run_id
    else:
        parent_run_id = parent_runs.iloc[0].run_id

    return {
        "parent_run_id": parent_run_id,
        "run_name": dvc_exp_name,
        "nested": True,
    }


def train_model(
    train_data: pd.DataFrame,
    params: dict[str, int | float],
) -> str:
    experiment = mlflow.set_experiment("ml_classification")
    mlflow.keras.autolog()

    run_args = _mlflow_run_arguments(experiment.experiment_id)
    training_params = dict(params)
    random_seed = int(training_params["random_seed"])

    with mlflow.start_run(**run_args) as run:
        mlflow.set_tag("pipeline_stage", "train")
        mlflow.log_params(training_params)
        tf.keras.utils.set_random_seed(random_seed)

        mlflow.log_artifact("artifacts/[features]_mean_imputer.joblib")
        mlflow.log_artifact("artifacts/[features]_scaler.joblib")

        X_train, y_train, encoder = prepare_data(train_data)
        model = create_model(
            input_shape=X_train.shape[1],
            num_classes=y_train.shape[1],
            params=training_params,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        logger.info("Training model...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=int(training_params["epochs"]),
            batch_size=int(training_params["batch_size"]),
            callbacks=[early_stopping],
        )

        save_training_artifacts(model, encoder)
        mlflow.log_artifact("artifacts/[target]_one_hot_encoder.joblib")

        val_accuracy = history.history.get("val_accuracy", [])
        best_val_accuracy = max(val_accuracy) if val_accuracy else None
        if best_val_accuracy is not None:
            mlflow.log_metric("selection_val_accuracy", float(best_val_accuracy))

        metrics = {
            metric: float(values[-1])
            for metric, values in history.history.items()
            if values
        }
        if best_val_accuracy is not None:
            metrics["best_val_accuracy"] = float(best_val_accuracy)

        with open("metrics/training.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        run_metadata = {
            "run_id": run.info.run_id,
            "experiment_id": experiment.experiment_id,
        }
        with open("artifacts/training_run.json", "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2)

        return run.info.run_id


def main() -> None:
    train_data = load_data()
    params = load_params()
    run_id = train_model(train_data, params)
    logger.info("Model training completed in MLflow run %s", run_id)


if __name__ == "__main__":
    main()
