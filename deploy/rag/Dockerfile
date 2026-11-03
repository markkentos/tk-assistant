FROM python:3.11-slim

WORKDIR /app

# torch — CPU-сборка (190 МБ); модели эмбеддера/переранкинщика монтируются
# с хоста через volume (см. docker-compose.yml), а не качаются в образ
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_service.py .

EXPOSE 8000
CMD ["uvicorn", "rag_service:app", "--host", "0.0.0.0", "--port", "8000"]
