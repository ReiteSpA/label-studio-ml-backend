"""
Tests for the RF-DETR ML backend API.

Run with:
    pip install -r requirements-test.txt
    pytest test_api.py -v
"""

import json
import pytest


@pytest.fixture
def client():
    from _wsgi import init_app
    from model import RFDETRBackend
    app = init_app(model_class=RFDETRBackend)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


LABEL_CONFIG = (
    '<View>'
    '<Image name="image" value="$image"/>'
    '<RectangleLabels name="label" toName="image">'
    '<Label value="person" background="green"/>'
    '<Label value="car" background="blue"/>'
    '</RectangleLabels>'
    '</View>'
)


def test_health(client):
    """Server should expose a health or root endpoint."""
    response = client.get("/health")
    if response.status_code == 404:
        response = client.get("/")
    assert response.status_code in (200, 404)


def test_predict_empty_result_on_missing_image(client):
    """With a missing image path, predict should return 200 and empty or valid results."""
    request = {
        "tasks": [{"id": 1, "data": {"image": "/nonexistent/image.jpg"}}],
        "label_config": LABEL_CONFIG,
    }
    response = client.post(
        "/predict",
        data=json.dumps(request),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 1
    # Missing file → backend typically returns empty result
    assert "result" in data["results"][0]
    assert isinstance(data["results"][0]["result"], list)
