FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

CMD ["python", "-m", "online.api", "--config", "config/settings.yaml", "--host", "0.0.0.0", "--port", "8000"]
