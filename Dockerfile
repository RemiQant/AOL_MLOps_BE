FROM python:3.11-slim
# LightGBM butuh OpenMP, tidak ada di image slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY api/ api/
COPY data/dataset/air_quality_historical.csv data/dataset/
COPY notebooks/models/model.joblib notebooks/models/comparison.json notebooks/models/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]