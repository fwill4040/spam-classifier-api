from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import pipeline
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spam Classifier API",
    description="MLOps portfolio project — classifies text as spam or ham",
    version="1.0.0"
)

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total number of predictions made",
    ["label"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing a prediction",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

SPAM_DETECTED = Counter(
    "spam_detected_total",
    "Total number of spam messages detected"
)

# Load model once at startup
logger.info("Loading model...")
classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    tokenizer="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)
logger.info("Model loaded successfully.")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    is_spam: bool


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "spam-classifier-v1"}


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logger.info(f"Classifying text of length {len(request.text)}")

    start_time = time.time()
    result = classifier(request.text)[0]
    latency = time.time() - start_time

    label = result["label"]
    confidence = result["score"]
    is_spam = label == "LABEL_1"

    # Record metrics
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_COUNT.labels(label="spam" if is_spam else "ham").inc()
    if is_spam:
        SPAM_DETECTED.inc()

    logger.info(f"Prediction: {'spam' if is_spam else 'ham'} in {latency:.3f}s")

    return PredictResponse(
        text=request.text,
        label="spam" if is_spam else "ham",
        confidence=round(confidence, 4),
        is_spam=is_spam
    )


@app.get("/")
def root():
    return {
        "message": "Spam Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "metrics": "GET /metrics"
    }