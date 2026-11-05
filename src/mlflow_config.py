import os

from dotenv import load_dotenv


def configure_mlflow_tracking() -> str | None:
    """Configure remote MLflow tracking only when explicitly available.

    A preconfigured MLFLOW_TRACKING_URI always wins. If no URI is set but a
    DAGSHUB_USER_TOKEN is available, DagsHub integration is initialized for
    this repository. Otherwise MLflow keeps its local default tracking store.
    """
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri

    if os.getenv("DAGSHUB_USER_TOKEN"):
        import dagshub

        dagshub.init(
            repo_owner="publiomcko-cloud",
            repo_name="mlops_project",
            mlflow=True,
        )
        return os.getenv("MLFLOW_TRACKING_URI")

    return None
