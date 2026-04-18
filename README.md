\# Spam Classifier API



An MLOps portfolio project demonstrating end-to-end deployment of a machine learning model.



!\[CI/CD](https://github.com/YOUR\_GITHUB\_USERNAME/spam-classifier-api/actions/workflows/ci-cd.yml/badge.svg)



\## Stack

\- \*\*Model\*\*: BERT-tiny fine-tuned for SMS spam detection (Hugging Face)

\- \*\*API\*\*: FastAPI with automatic OpenAPI docs

\- \*\*Container\*\*: Docker with multi-layer caching

\- \*\*CI/CD\*\*: GitHub Actions — test → build → push to registry



\## Architecture

Every `git push` to main triggers:

1\. Automated tests via pytest

2\. Docker image build

3\. Push to GitHub Container Registry (only if tests pass)



\## Run locally



```bash

docker pull ghcr.io/YOUR\_GITHUB\_USERNAME/spam-classifier-api:latest

docker run -p 8000:8000 ghcr.io/YOUR\_GITHUB\_USERNAME/spam-classifier-api:latest

```



Then open http://localhost:8000/docs for the interactive API explorer.



\## Example request



```bash

curl -X POST http://localhost:8000/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"text": "You have won a FREE iPhone! Click here to claim!"}'

```



\## Example response



```json

{

&#x20; "text": "You have won a FREE iPhone! Click here to claim!",

&#x20; "label": "spam",

&#x20; "confidence": 0.9987,

&#x20; "is\_spam": true

}

```



\## Run tests locally



```bash

pip install -r requirements.txt

pytest tests/ -v

```

