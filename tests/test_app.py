from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_spam():
    payload = {"text": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_spam"] == True
    assert "confidence" in data


def test_predict_ham():
    payload = {"text": "Hey, are we still meeting for lunch tomorrow?"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_spam"] == False


def test_empty_text_returns_400():
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_root():
    response = client.get("/")
    assert response.status_code == 200