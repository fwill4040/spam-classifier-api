# Spam Classifier API

An MLOps portfolio project demonstrating end-to-end deployment of a machine learning model.

![CI/CD](https://github.com/fwill4040/spam-classifier-api/actions/workflows/ci-cd.yml/badge.svg)

## Live Demo
API is live at: https://spam-classifier-api-9cfe.onrender.com

Interactive docs: https://spam-classifier-api-9cfe.onrender.com/docs

## Live Monitoring Dashboard
![Grafana Dashboard](grafana-dashboard.png)

## Stack
- **Model**: BERT-tiny fine-tuned for SMS spam detection (Hugging Face)
- **API**: FastAPI with automatic OpenAPI docs
- **Container**: Docker with multi-layer caching
- **CI/CD**: GitHub Actions — test → build → push to registry
- **Monitoring**: Prometheus metrics + Grafana dashboard
- **Deployment**: Render.com — live public HTTPS endpoint

## Metrics tracked
- Total predictions (spam vs ham)
- Average prediction latency (seconds)
- Spam detection rate (%)
- Prediction rate per minute

## Architecture
Every `git push` to main triggers:
1. Automated tests via pytest
2. Docker image build
3. Push to GitHub Container Registry (only if tests pass)
4. Auto-deploy to Render

## Example request

```bash
curl -X POST https://spam-classifier-api-9cfe.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You have won a FREE iPhone! Click here to claim!"}'
```

## Example response

```json
{
  "text": "You have won a FREE iPhone! Click here to claim!",
  "label": "spam",
  "confidence": 0.9987,
  "is_spam": true
}
```

## Run locally

```bash
git clone https://github.com/fwill4040/spam-classifier-api.git
cd spam-classifier-api
pip install -r requirements.txt
uvicorn app:app --reload
```

## Run tests

```bash
pytest tests/ -v
```