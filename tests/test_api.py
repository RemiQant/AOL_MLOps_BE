from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_latest():
    body = client.get("/predict").json()
    assert body["actual"] is None
    assert len(body["history"]) == 30
    assert body["pm25"] > 1  # kalau ~0, pm25_today lupa ditambahkan


def test_time_machine():
    body = client.get("/predict", params={"date": "2026-01-10"}).json()
    assert body["actual"] is not None and body["category_match"] is not None


def test_bad_date():
    assert client.get("/predict", params={"date": "1999-01-01"}).status_code == 400
    assert client.get("/predict", params={"date": "abc"}).status_code == 422


def test_monitoring_simulated():
    body = client.get("/monitoring", params={"simulate_drift": True}).json()
    assert len(body["drift"]) == 5 and body["simulated"] is True


def test_models_and_reload():
    body = client.get("/models").json()
    assert body["served"] in [m["name"] for m in body["models"]]
    assert client.post("/reload").json()["status"] == "reloaded"