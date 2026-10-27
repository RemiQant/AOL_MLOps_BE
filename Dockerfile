FROM python:3.13-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# Serving only: training code and torch/mlflow stay out of the image
COPY src/api.py src/data.py ./src/
COPY dataset/air_quality_historical.csv ./dataset/
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
