FROM python:3.11-slim

WORKDIR /app

# system deps for Pillow / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HYBRIDPROP_CONFIG=ml/config/config.yaml
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
