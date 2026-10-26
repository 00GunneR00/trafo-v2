# Tahmin + alarm servisi. GPU gerekmez: tek pencere için CPU çıkarımı milisaniyeler sürer.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY reports/anomaly.json reports/anomaly.json
COPY data/raw/ETTh1.csv data/raw/ETTh1.csv

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
