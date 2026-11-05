import yaml
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


project_root = Path(__file__).resolve().parents[1]

def get_dvc_stages():
    dvc_yaml_path = project_root / "dvc.yaml"
    with open(dvc_yaml_path) as f:
        dvc_config = yaml.safe_load(f)
    return list(dvc_config["stages"].keys())

def register_artifacts_callable():
    from src.register_artifacts import main
    main()

default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    "ml_pipeline",
    default_args=default_args
) as dag:
    # DVC Pipeline Stages
    dvc_stages = get_dvc_stages()

    # Create tasks for each DVC stage
    dvc_tasks = []
    for stage in dvc_stages:
        task = BashOperator(
            task_id=f"dvc_{stage}",
            cwd=project_root,
            bash_command=f"dvc repro {stage}"
        )
        dvc_tasks.append(task)

    # Register artifacts in MLflow
    register_artifacts = PythonOperator(
        task_id="register_artifacts",
        python_callable=register_artifacts_callable
    )

    # Deploy model by building and running Docker container
    create_app_image = BashOperator(
        task_id="create_app_image",
        cwd=project_root,
        bash_command = "docker build -t ml-classifier ."
    )

    # Set dependencies
    for i in range(len(dvc_tasks) - 1):
        dvc_tasks[i] >> dvc_tasks[i + 1]

    # Connect the last DVC task to register_artifacts and then to create_app_image
    dvc_tasks[-1] >> register_artifacts >> create_app_image
