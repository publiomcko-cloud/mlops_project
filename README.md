# ML Classifier - Reproducible MLOps Portfolio Project

[![CI](https://github.com/publiomcko-cloud/mlops_project/actions/workflows/ci.yml/badge.svg)](https://github.com/publiomcko-cloud/mlops_project/actions/workflows/ci.yml)

A reproducible machine-learning project that trains, evaluates, tracks, packages, and serves a neural-network classifier using the scikit-learn breast cancer dataset.

The project is designed as an MLOps portfolio demonstration rather than only a notebook or isolated model-training exercise. It includes a staged DVC pipeline, MLflow experiment tracking and model registration, deterministic preprocessing, a Flask inference service, Docker packaging, automated tests, and GitHub Actions CI.

> Portfolio and educational use only. This project is not a medical device and must not be used for clinical diagnosis, treatment, or patient decision-making.

## What this project demonstrates

- Reproducible ML pipelines with DVC
- Experiment and artifact tracking with MLflow
- Deterministic train/test preprocessing boundaries
- TensorFlow/Keras model training
- Validation-based model selection
- Held-out test evaluation
- Model registry integration
- Flask inference service
- Containerization with Docker and Gunicorn
- Health and readiness endpoints
- Upload validation and bounded inference inputs
- Automated tests and CI

## Architecture

~~~mermaid
flowchart LR
    A[scikit-learn dataset] --> B[DVC load_data]
    B --> C[Train/Test Split]
    C --> D[Train-fitted Imputer]
    D --> E[Train-fitted StandardScaler]
    E --> F[TensorFlow Training]
    F --> G[MLflow Run]
    F --> H[Validation Metrics]
    G --> I[Held-out Test Evaluation]
    H --> J[Model Selection]
    I --> K[Test Metrics]
    J --> L[MLflow Model Registry]
    F --> M[Local DVC Artifacts]
    M --> N[Flask Inference Service]
    L --> N
    N --> O[Gunicorn / Docker]
~~~

## Evaluation methodology

The project separates model selection from final test evaluation.

1. The raw dataset is split into training and held-out test sets before preprocessing.
2. Imputation and scaling are fitted only on the training partition.
3. Training uses an internal validation split.
4. Validation accuracy is recorded as the model-selection signal.
5. The held-out test set is evaluated only after training.
6. Test accuracy is recorded as final evaluation evidence and is not used to choose the registered model.

This avoids selecting a model directly against the test set.

## Project structure

~~~text
mlops_project/
├── .github/workflows/ci.yml
├── app/
│   ├── main.py
│   └── templates/index.html
├── artifacts/
├── data/
│   ├── raw/
│   ├── preprocessed/
│   └── processed/
├── metrics/
├── models/
├── src/
│   ├── data_loading/
│   ├── data_preprocessing/
│   ├── feature_engineering/
│   ├── model_training/
│   ├── model_evaluation/
│   └── register_artifacts.py
├── tests/
│   ├── test_app.py
│   └── test_pipeline.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── Dockerfile
└── pyproject.toml
~~~

## Requirements

- Python 3.12+
- Git
- Optional: Docker
- Optional: access to the configured DVC remote for restoring generated artifacts

## Installation

~~~bash
git clone https://github.com/publiomcko-cloud/mlops_project.git
cd mlops_project

python -m venv .venv
~~~

Linux/macOS:

~~~bash
source .venv/bin/activate
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Install the project and development dependencies:

~~~bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
~~~

Apache Airflow is optional and intentionally excluded from the default runtime dependencies:

~~~bash
pip install -e ".[orchestration]"
~~~

## Reproduce the ML pipeline

The complete pipeline is declared in dvc.yaml.

~~~bash
dvc repro
~~~

Equivalent stages are:

~~~bash
python -m src.data_loading.load_data
python -m src.data_preprocessing.preprocess_data
python -m src.feature_engineering.engineer_features
python -m src.model_training.train_model
python -m src.model_evaluation.evaluate_model
~~~

Inspect DVC metrics with:

~~~bash
dvc metrics show
~~~

The training stage writes explicit MLflow run metadata to artifacts/training_run.json. The evaluation stage uses that exact run identifier instead of guessing the latest MLflow run.

## Model registration

Registration is intentionally separate from test evaluation.

~~~bash
python -m src.register_artifacts
~~~

The registration code selects an evaluated run using validation accuracy, not test accuracy.

## DVC artifacts

Generated datasets, preprocessing artifacts, metrics, and trained models are not committed directly to Git.

If the configured remote contains the latest artifacts:

~~~bash
dvc pull
~~~

The repository DVC configuration points to the project remote on DagsHub. Credentials, when required by the remote or MLflow tracking server, must be supplied through local environment configuration and must never be committed.

## MLflow

Without an external tracking URI, MLflow can use its local tracking store.

For a remote tracking server, configure the environment before training or inference, for example:

~~~bash
export MLFLOW_TRACKING_URI=https://example-mlflow-server
~~~

For registry-based inference, the application accepts:

~~~bash
export MODEL_SOURCE=mlflow
export MLFLOW_MODEL_URI=models:/model/latest
~~~

MODEL_SOURCE supports:

- auto: prefer local DVC artifacts, otherwise try MLflow
- local: require local model and preprocessing artifacts
- mlflow: require the configured MLflow model URI

## Run the web application

Restore or generate artifacts first:

~~~bash
dvc pull
~~~

or:

~~~bash
dvc repro
~~~

Then start Flask:

~~~bash
python app/main.py
~~~

Open:

~~~text
http://localhost:5001
~~~

Operational endpoints:

- GET /health returns process health independently of model availability.
- GET /ready returns HTTP 200 only when the inference model is available.
- POST /upload accepts a CSV file for prediction.

The service can stay observable even if model loading fails. In that state /health remains available while /ready returns HTTP 503.

## Input controls

The inference endpoint enforces:

- CSV extension validation
- UTF-8 parsing
- exact required feature validation
- a default 2 MB upload limit
- a default maximum of 10,000 prediction rows
- generic client-facing errors for unexpected failures

The limits can be configured with:

~~~bash
export MAX_UPLOAD_BYTES=2097152
export MAX_PREDICTION_ROWS=10000
~~~

## Docker

The Docker image can include local DVC-generated model artifacts. Restore or generate them before building:

~~~bash
dvc pull
docker build -t ml-classifier .
docker run --rm -p 5001:5001 ml-classifier
~~~

If the image is built without model artifacts, the web process still starts for observability, but /ready reports the service as unavailable until a usable model source is configured.

The container includes a health check against /health and serves the application through Gunicorn.

## Tests

Run locally with:

~~~bash
python -m pytest
~~~

The current test suite covers:

- process health versus model readiness
- valid CSV prediction requests
- missing feature rejection
- prediction row limits
- suppression of internal error details
- train-only imputer fitting
- train-only scaler fitting
- DVC training/evaluation run linkage

## Continuous integration

GitHub Actions runs on pushes to main, feature/fix branches, and pull requests targeting main.

CI performs:

- Python source compilation
- DVC YAML parsing
- automated tests

The CI dependency set is intentionally lightweight so routine source validation does not need to install TensorFlow or run model training.

## Model and dataset

Dataset: scikit-learn Breast Cancer Wisconsin Diagnostic dataset.

Model: feed-forward neural network with two hidden layers and dropout. Hyperparameters are declared in params.yaml.

The dataset is suitable for demonstrating an ML engineering workflow because it is deterministic, public, small enough for reproducible development, and requires no private patient data.

## Reproducibility notes

- Dataset perturbation uses a fixed seed.
- Train/test splitting uses a configured random seed.
- TensorFlow random state is configured from params.yaml.
- DVC tracks pipeline dependencies and outputs.
- MLflow records parameters, validation metrics, test metrics, and model artifacts.
- Evaluation is attached to the exact training run through persisted run metadata.

## Scope

This repository demonstrates MLOps engineering practices on a compact educational dataset. It is not intended to claim clinical validity, production medical approval, or real-world diagnostic performance.
