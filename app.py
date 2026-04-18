from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spam Classifier API",
    description="MLOps portfolio project — classifies text as spam or ham",
    version="1.0.0"
)

# Load model once at startup — this is key MLOps practice
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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logger.info(f"Classifying text of length {len(request.text)}")
    result = classifier(request.text)[0]

    label = result["label"]
    confidence = result["score"]
    is_spam = label == "LABEL_1"

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
        "predict": "POST /predict"
    }