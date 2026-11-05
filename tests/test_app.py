import io

import pandas as pd
from sklearn.datasets import load_breast_cancer

from app.main import create_app


class FakeModelService:
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"Prediction": [1] * len(features)},
            index=features.index,
        )


class FailingModelService:
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("internal-secret-detail")


def make_csv(rows: int = 1) -> bytes:
    dataset = load_breast_cancer()
    frame = pd.DataFrame(
        [dataset.data[0]] * rows,
        columns=dataset.feature_names,
    )
    return frame.to_csv(index=False).encode("utf-8")


def test_health_is_independent_from_model_readiness():
    app = create_app(load_model=False)
    app.config["TESTING"] = True
    client = app.test_client()

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503


def test_ready_with_injected_model_service():
    app = create_app(model_service=FakeModelService())
    app.config["TESTING"] = True

    response = app.test_client().get("/ready")

    assert response.status_code == 200
    assert response.get_json()["model_ready"] is True


def test_upload_accepts_valid_feature_csv():
    app = create_app(model_service=FakeModelService())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(make_csv()), "sample.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data


def test_upload_rejects_missing_features():
    app = create_app(model_service=FakeModelService())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"mean radius\n1.0\n"), "sample.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Missing required columns" in response.data


def test_upload_enforces_row_limit():
    app = create_app(model_service=FakeModelService())
    app.config["TESTING"] = True
    app.config["MAX_PREDICTION_ROWS"] = 1
    client = app.test_client()

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(make_csv(rows=2)), "sample.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"maximum of 1 rows" in response.data


def test_internal_prediction_errors_are_not_exposed():
    app = create_app(model_service=FailingModelService())
    app.config["TESTING"] = False
    client = app.test_client()

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(make_csv()), "sample.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 500
    assert b"The prediction could not be completed." in response.data
    assert b"internal-secret-detail" not in response.data
