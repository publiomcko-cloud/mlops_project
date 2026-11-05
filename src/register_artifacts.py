import logging

import mlflow
from mlflow.tracking import MlflowClient

from src.mlflow_config import configure_mlflow_tracking


logger = logging.getLogger("src.register_artifacts")


def select_model_run(client: MlflowClient, experiment_id: str):
    """Select a model using validation evidence only."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.pipeline_stage = 'evaluated'",
        order_by=[
            "metrics.selection_val_loss ASC",
            "start_time DESC",
        ],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            "No evaluated training run with validation metrics is available."
        )
    return runs[0]


def register_model() -> str:
    configure_mlflow_tracking()
    client = MlflowClient()

    experiment = client.get_experiment_by_name("ml_classification")
    if experiment is None:
        raise RuntimeError(
            "MLflow experiment 'ml_classification' does not exist. "
            "Run the training pipeline first."
        )

    selected_run = select_model_run(client, experiment.experiment_id)
    run_id = selected_run.info.run_id
    selection_score = selected_run.data.metrics.get("selection_val_loss")

    logger.info(
        "Registering run %s selected by validation loss %s",
        run_id,
        selection_score,
    )

    try:
        client.create_registered_model("model")
    except mlflow.exceptions.MlflowException:
        logger.debug("Registered model already exists")

    model_uri = f"runs:/{run_id}/model"
    version = client.create_model_version(
        name="model",
        source=model_uri,
        run_id=run_id,
    )
    logger.info("Registered model version %s", version.version)
    return str(version.version)


def main() -> None:
    version = register_model()
    logger.info("Model registration completed: version %s", version)


if __name__ == "__main__":
    main()
