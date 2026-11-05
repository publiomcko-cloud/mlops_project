import io
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer


logger = logging.getLogger("app.main")

DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_PREDICTION_ROWS = 10_000


class ModelService:
    """Load the model and preprocessing artifacts for inference.

    Auto mode prefers local DVC artifacts and falls back to an MLflow model.
    Set MODEL_SOURCE to local or mlflow to force one mode.
    """

    def __init__(self) -> None:
        self.source = os.getenv("MODEL_SOURCE", "auto").lower()
        if self.source not in {"auto", "local", "mlflow"}:
            raise ValueError("MODEL_SOURCE must be one of: auto, local, mlflow")
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if self.source in {"auto", "local"} and self._local_artifacts_available():
            self._load_local_artifacts()
            return

        if self.source == "local":
            raise RuntimeError(
                "Local model artifacts are unavailable. Run dvc pull or dvc repro."
            )

        self._load_mlflow_artifacts()

    @staticmethod
    def _artifact_paths() -> dict[str, Path]:
        return {
            "model": Path(os.getenv("MODEL_PATH", "models/model.keras")),
            "imputer": Path(
                os.getenv(
                    "IMPUTER_PATH",
                    "artifacts/[features]_mean_imputer.joblib",
                )
            ),
            "scaler": Path(
                os.getenv("SCALER_PATH", "artifacts/[features]_scaler.joblib")
            ),
            "encoder": Path(
                os.getenv(
                    "ENCODER_PATH",
                    "artifacts/[target]_one_hot_encoder.joblib",
                )
            ),
        }

    def _local_artifacts_available(self) -> bool:
        return all(path.is_file() for path in self._artifact_paths().values())

    def _load_local_artifacts(self) -> None:
        import tensorflow as tf

        paths = self._artifact_paths()
        logger.info("Loading model and preprocessors from local artifacts")
        self.model = tf.keras.models.load_model(paths["model"])
        self.features_imputer = joblib.load(paths["imputer"])
        self.features_scaler = joblib.load(paths["scaler"])
        self.target_encoder = joblib.load(paths["encoder"])

    def _load_mlflow_artifacts(self) -> None:
        from src.mlflow_config import configure_mlflow_tracking

        configure_mlflow_tracking()
        import mlflow

        model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/model/latest")
        logger.info("Loading model from MLflow URI %s", model_uri)
        self.model = mlflow.keras.load_model(model_uri)

        model_info = mlflow.models.get_model_info(model_uri)
        run_id = model_info.run_id
        if not run_id:
            raise RuntimeError(
                "The configured MLflow model URI does not expose a source run ID."
            )

        artifacts_dir = Path(
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
        )
        self.features_imputer = joblib.load(
            artifacts_dir / "[features]_mean_imputer.joblib"
        )
        self.features_scaler = joblib.load(
            artifacts_dir / "[features]_scaler.joblib"
        )
        self.target_encoder = joblib.load(
            artifacts_dir / "[target]_one_hot_encoder.joblib"
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)
        y_pred = self.model.predict(X_scaled, verbose=0)
        y_decoded = self.target_encoder.inverse_transform(y_pred)
        return pd.DataFrame(
            {"Prediction": y_decoded.ravel()},
            index=features.index,
        )


def create_routes(app: Flask) -> None:
    @app.route("/")
    def index() -> str:
        return render_template(
            "index.html",
            model_ready=app.model_service is not None,
        )

    @app.route("/health")
    def health():
        return {"status": "ok"}, 200

    @app.route("/ready")
    def ready():
        if app.model_service is None:
            return {"status": "unavailable", "model_ready": False}, 503
        return {"status": "ready", "model_ready": True}, 200

    @app.route("/upload", methods=["POST"])
    def upload():
        if app.model_service is None:
            return (
                render_template(
                    "index.html",
                    error=(
                        "The prediction model is not available. "
                        "Load artifacts with dvc pull or configure MLflow."
                    ),
                    model_ready=False,
                ),
                503,
            )

        file = request.files.get("file")
        if file is None or not file.filename:
            return render_template(
                "index.html",
                error="Please choose a CSV file.",
                model_ready=True,
            )

        if not file.filename.lower().endswith(".csv"):
            return render_template(
                "index.html",
                error="Please upload a CSV file.",
                model_ready=True,
            )

        try:
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            max_rows = app.config["MAX_PREDICTION_ROWS"]
            if len(features) > max_rows:
                return render_template(
                    "index.html",
                    error=f"CSV exceeds the maximum of {max_rows} rows.",
                    model_ready=True,
                )

            expected_features = list(load_breast_cancer().feature_names)
            missing_cols = [
                col for col in expected_features if col not in features.columns
            ]
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                    model_ready=True,
                )

            features = features[expected_features]
            predictions = app.model_service.predict(features)

            return render_template(
                "index.html",
                predictions=predictions.to_string(),
                model_ready=True,
            )
        except (UnicodeDecodeError, pd.errors.ParserError):
            return render_template(
                "index.html",
                error="The uploaded file is not a valid UTF-8 CSV.",
                model_ready=True,
            )
        except Exception:
            logger.exception("Unexpected error while processing prediction upload")
            return (
                render_template(
                    "index.html",
                    error="The prediction could not be completed.",
                    model_ready=True,
                ),
                500,
            )


def create_app(
    model_service: ModelService | None = None,
    *,
    load_model: bool = True,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.getenv("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)
    )
    app.config["MAX_PREDICTION_ROWS"] = int(
        os.getenv("MAX_PREDICTION_ROWS", DEFAULT_MAX_PREDICTION_ROWS)
    )

    app.model_service = model_service
    if model_service is None and load_model:
        try:
            app.model_service = ModelService()
        except Exception:
            logger.exception("Model initialization failed; serving in unready mode")
            app.model_service = None

    create_routes(app)
    return app


app = create_app()


def main() -> None:
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
