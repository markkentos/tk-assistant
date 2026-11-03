FROM python:3.11-slim

WORKDIR /app

# torch — CPU-сборка (190 МБ вместо 2+ ГБ CUDA-колеса)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка эмбеддера в образ: сервис стартует без доступа в интернет
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY rag_service.py .

EXPOSE 8000
CMD ["uvicorn", "rag_service:app", "--host", "0.0.0.0", "--port", "8000"]
